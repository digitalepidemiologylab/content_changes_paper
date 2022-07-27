import datetime
import time
import math
import tqdm

import argparse
import os
import json

import sys
# Import from official repo
sys.path.append('tensorflow_models')
from official.utils.misc import distribution_utils
from official.modeling import performance
from official.utils.misc import keras_utils

import input_pipeline_multinomial as input_pipeline 
import bert_models_multinomial as bert_models

from transformers import PretrainedConfig
# from config import PRETRAINED_MODELS
from utils.misc import ArgParseDefault, save_to_json, add_bool_arg
from utils.finetune_helpers import Metrics
import utils.optimizer

import tensorflow as tf
# Remove duplicate logger (not sure why this is happening, possibly an issue with the imports)
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

# Add file logging
import logging
from logging.handlers import RotatingFileHandler
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)
handler = RotatingFileHandler("logs/finetune.log", maxBytes=2000, backupCount=10)
logger.addHandler(handler)

def get_model_config(config_path):
    config = PretrainedConfig.from_json_file(config_path)
    return config

def get_input_meta_data(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, 'meta.json'), 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))
    return input_meta_data

def configure_optimizer(optimizer, use_float16=False, use_graph_rewrite=False, loss_scale='dynamic'):
    """Configures optimizer object with performance options."""
    if use_float16:
        # Wraps optimizer with a LossScaleOptimizer. This is done automatically in compile() with the
        # "mixed_float16" policy, but since we do not call compile(), we must wrap the optimizer manually.
        optimizer = (tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale=loss_scale))
    if use_graph_rewrite:
        # Note: the model dtype must be 'float32', which will ensure
        # tf.keras.mixed_precision and tf.train.experimental.enable_mixed_precision_graph_rewrite do not double up.
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    return optimizer

def get_model(args, model_config, model_path, steps_per_epoch, warmup_steps, num_labels, max_seq_length):
    # Get classifier and core model (used to initialize from checkpoint)
    classifier_model, core_model = bert_models.classifier_model(
            model_config,
            model_path,
            num_labels,
            max_seq_length)
    # Optimizer
    optimizer = utils.optimizer.create_optimizer(
            args.learning_rate,
            steps_per_epoch * args.num_epochs,
            warmup_steps,
            args.end_lr,
            args.weight_decay_rate,
            args.optimizer_type)
    classifier_model.optimizer = configure_optimizer(
            optimizer,
            use_float16=False,
            use_graph_rewrite=False)
    return classifier_model, core_model

def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training=True):
  """Gets a closure to create a dataset."""
  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_classifier_dataset(
        input_file_pattern,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return dataset

  return _dataset_fn

def get_loss_fn(num_classes):
    """Gets the classification loss function."""
    def classification_loss_fn(labels, logits): # logits is a 2D tensor (each row corresponds to a sentence)
        """Classification loss."""
        # labels is a 1D tensor filled with numerically encoded labels (each entry corresponds to a sentence)
        labels = tf.squeeze(labels)
        # one_hot_labels is a 2D tensor (each row corresponds to a sentence)
        one_hot_labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        
        # log_probs is a 2D tensor (each row corresponds to a sentence)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        # Below: one_hot_labels * log_probs is a 2D tensor; the reduce_sum operation along the second dimension yields a 1D tensor (length = size of minibatch)

        # Elementwise multiplication between two 1D tensors; each element corresponds to the cross-entropy loss of a particular example 
        per_example_loss = -tf.reduce_sum(tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
        # Average loss in the minibatch (average_loss is a scalar)
        average_loss = tf.reduce_mean(per_example_loss, axis=-1)
        return average_loss
    return classification_loss_fn

def get_label_mapping(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, 'label_mapping.json'), 'rb') as reader:
        label_mapping = json.loads(reader.read().decode('utf-8'))
    label_mapping = dict(zip(range(len(label_mapping)), label_mapping))
    return label_mapping


def get_metrics():
    return [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]

def get_run_name(args):
    # Use timestamp to generate a unique run name
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    if args.run_prefix:
        run_name = f'run_{ts}_{args.run_prefix}'
    else:
        run_name = f'run_{ts}'
    return run_name

def run(args):
    """Train using the Keras/TF 2.0. Adapted from the tensorflow/models Github"""
    # CONFIG
    run_name = get_run_name(args)
    logger.info(f'*** Starting run {run_name} ***')
    data_dir = f'gs://{args.bucket_name}/data/finetune_multinomial/finetune_data/{args.finetune_data}'
    output_dir = f'gs://{args.bucket_name}/data/finetune_multinomial/runs/{run_name}'

    # Get configs
    pretrained_model_config_path = os.path.join('configs','bert_config_covid_twitter_bert.json')
    model_config = get_model_config(pretrained_model_config_path)
    # Path to the pretrained model on www.huggingface.co/
    model_path = 'digitalepidemiologylab/covid-twitter-bert-v2' 
    
    # Meta data
    input_meta_data = get_input_meta_data(data_dir)
    logger.info(f'Loaded training data meta.json file: {input_meta_data}')
    # Label mapping
    label_mapping = get_label_mapping(data_dir)

    # Calculate steps, warmup steps and eval steps
    train_data_size = input_meta_data['train_data_size']
    num_labels = input_meta_data['num_labels']
    max_seq_length = input_meta_data['max_seq_length']
    if args.limit_train_steps is None:
        steps_per_epoch = int(train_data_size / args.train_batch_size)
    else:
        steps_per_epoch = args.limit_train_steps
    warmup_steps = int(args.num_epochs * train_data_size * args.warmup_proportion / args.train_batch_size)
    if args.limit_eval_steps is None:
        eval_steps = int(math.floor(input_meta_data['eval_data_size'] / args.eval_batch_size))
    else:
        eval_steps = args.limit_eval_steps

    # Some logging
    logger.info(f'Finetuning on dataset {args.finetune_data}')
    logger.info(f'Running {args.num_epochs} epochs with {steps_per_epoch:,} steps per epoch')
    logger.info(f'Using warmup proportion of {args.warmup_proportion}, resulting in {warmup_steps:,} warmup steps')
    logger.info(f'Using learning rate: {args.learning_rate}, training batch size: {args.train_batch_size}, num_epochs: {args.num_epochs}')

    # Get model
    classifier_model, core_model = get_model(args, model_config, model_path, steps_per_epoch, warmup_steps, num_labels, max_seq_length)
    optimizer = classifier_model.optimizer
    loss_fn = get_loss_fn(num_labels)
    try:
        if ',' in args.validation_freq:
            validation_freq = args.validation_freq.split(',')
            validation_freq = [int(v) for v in validation_freq]
        else:
            validation_freq = int(args.validation_freq)
    except:
        raise ValueError(f'Invalid argument for validation_freq!')
    logger.info(f'Using a validation frequency of {validation_freq}')

    # Restore checkpoint
    if args.init_checkpoint:
        checkpoint_path = f'gs://{args.bucket_name}/pretrain/runs/{args.init_checkpoint}'
        checkpoint = tf.train.Checkpoint(model=core_model)
        checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
        logger.info(f'Successfully restored checkpoint from {checkpoint_path}')

    # Run Keras compile
    logger.info(f'Compiling Keras model...')
    classifier_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=get_metrics())
    logger.info(f'... done')

    # Create all custom callbacks
    summary_dir = os.path.join(output_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir, profile_batch=0)
    time_history_callback = keras_utils.TimeHistory(
        batch_size=args.train_batch_size,
        log_steps=args.time_history_log_steps,
        logdir=summary_dir)
    custom_callbacks = [summary_callback, time_history_callback]
    if args.save_model:
        logger.info('Using save_model option...')
        checkpoint_path = os.path.join(output_dir, 'model')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, save_best_only=True, verbose=1)
        custom_callbacks.append(checkpoint_callback)
    if args.early_stopping_epochs > 0:
        logger.info(f'Using early stopping of after {args.early_stopping_epochs} epochs of val_loss not decreasing')
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=args.early_stopping_epochs, monitor='val_loss')
        custom_callbacks.append(early_stopping_callback)

    # Generate dataset_fn
    train_input_fn = get_dataset_fn(
        os.path.join(data_dir, 'tfrecords', 'train.tfrecord'),
        max_seq_length,
        args.train_batch_size,
        is_training=True)
    eval_input_fn = get_dataset_fn(
        os.path.join(data_dir, 'tfrecords', 'dev.tfrecord'),
        max_seq_length,
        args.eval_batch_size,
        is_training=False)

    # Add metrics callback to calculate performance metrics at the end of epoch
    performance_metrics_callback = Metrics(
            eval_input_fn,
            label_mapping,
            os.path.join(summary_dir, 'metrics'),
            eval_steps,
            args.eval_batch_size,
            validation_freq)
    custom_callbacks.append(performance_metrics_callback)

    # Run keras fit
    time_start = time.time()
    logger.info('Run training...')
    history = classifier_model.fit(
        x=train_input_fn(),
        validation_data=eval_input_fn(),
        steps_per_epoch=steps_per_epoch,
        epochs=args.num_epochs,
        validation_steps=eval_steps,
        validation_freq=validation_freq,
        callbacks=custom_callbacks,
        verbose=1)
    time_end = time.time()
    training_time_min = (time_end-time_start)/60
    logger.info(f'Finished training after {training_time_min:.1f} min')

    # Write training log
    all_scores = performance_metrics_callback.scores
    all_predictions = performance_metrics_callback.predictions
    if len(all_scores) > 0:
        final_scores = all_scores[-1]
        logger.info(f'Final eval scores: {final_scores}')
    else:
        final_scores = {}
    full_history = history.history
    if len(full_history) > 0:
        final_val_loss = full_history['val_loss'][-1]
        final_loss = full_history['loss'][-1]
        logger.info(f'Final training loss: {final_loss:.2f}, Final validation loss: {final_val_loss:.2f}')
    else:
        final_val_loss = None
        final_loss = None
    data = {
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_name': run_name,
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'max_seq_length': max_seq_length,
            'num_train_steps': steps_per_epoch * args.num_epochs,
            'eval_steps': eval_steps,
            'steps_per_epoch': steps_per_epoch,
            'training_time_min': training_time_min,
            'data_dir': data_dir,
            'output_dir': output_dir,
            'all_scores': all_scores,
            'all_predictions': all_predictions,
            'num_labels': num_labels,
            'label_mapping': label_mapping,
            **full_history,
            **final_scores,
            **vars(args),
            }
    # Write run_log
    f_path_training_log = os.path.join(output_dir, 'run_logs.json')
    logger.info(f'Writing training log to {f_path_training_log}...')
    save_to_json(data, f_path_training_log)
    # Write BERT config
    model_config.id2label = label_mapping
    model_config.label2id = {v:k for k, v in label_mapping.items()}
    model_config.max_seq_length = max_seq_length
    model_config.num_labels = num_labels
    f_path_bert_config = os.path.join(output_dir, 'bert_config.json')
    logger.info(f'Writing BERT config to {f_path_bert_config}...')
    save_to_json(model_config.to_dict(), f_path_bert_config)

def set_mixed_precision_policy(args):
    """Sets mix precision policy."""
    if args.dtype == 'fp16':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale=loss_scale)
        tf.keras.mixed_precision.experimental.set_policy(policy)
    elif args.dtype == 'bf16':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    elif args.dtype == 'fp32':
        tf.keras.mixed_precision.experimental.set_policy('float32')
    else:
        raise ValueError(f'Unknown dtype {args.dtype}')

def main(args):
    # Set TF Hub caching to bucket
    os.environ['TFHUB_CACHE_DIR'] = os.path.join(f'gs://{args.bucket_name}/tmp')
    # Get distribution strategy
    if args.use_tpu:
        if args.tpu_ip:
            logger.info(f'Initializing TPU on address {args.tpu_ip}...')
            tpu_address = f'grpc://{args.tpu_ip}:8470'
            strategy = distribution_utils.get_distribution_strategy(distribution_strategy='tpu', tpu_address=tpu_address)
        elif args.tpu_name:
            logger.info(f'Initializing TPU with name {args.tpu_name}...')
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_name)
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        else:
            raise ValueError(f'You need to either specify a tpu_ip or a tpu_name in order to use a TPU.')
    else:
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='mirrored', num_gpus=args.num_gpus)
    # set mixed precision
    set_mixed_precision_policy(args)
    # Run training
    for repeat in range(args.repeats):
        with strategy.scope():
            run(args)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--finetune_data', required=True, help='Finetune data folder sub path. Path has to be in gs://{bucket_name}/data/finetune_multinomial/finetune_data/{finetune_data}.\
                    This folder includes a meta.json (containing meta info about the dataset) and a file label_mapping.json. \
                    TFRecord files (train.tfrecord and dev.tfrecord) should be located in a \
                    subfolder gs://{bucket_name}/data/finetune_multinomial/finetune_data/{finetune_data}/tfrecords/')
    parser.add_argument('--bucket_name', required=True, help='Bucket name')
    parser.add_argument('--tpu_ip', required=False, help='IP-address of the TPU')
    parser.add_argument('--tpu_name', required=False, help='Name of the TPU (required for pods)')
    parser.add_argument('--run_prefix', help='Prefix to be added to all runs. Useful to group runs')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--init_checkpoint', default=None, help='Run name to initialize checkpoint from. Example: "run2/ctl_step_8000.ckpt-8". \
            By default using a pretrained model from gs://{bucket_name}/pretrained_models/')
    parser.add_argument('--init_checkpoint_index', type=int, help='Checkpoint index. This argument is ignored and only added for reporting.')
    parser.add_argument('--repeats', default=1, type=int, help='Number of times the script should run.')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs')
    parser.add_argument('--limit_train_steps', type=int, help='Limit the number of train steps per epoch. Useful for testing.')
    parser.add_argument('--limit_eval_steps', type=int, help='Limit the number of eval steps per epoch. Useful for testing.')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='Eval batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--end_lr', default=0, type=float, help='Final learning rate')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Learning rate warmup proportion')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--early_stopping_epochs', default=-1, type=int, help='Stop when loss hasn\'t decreased during n epochs')
    parser.add_argument('--optimizer_type', default='adamw', choices=['adamw', 'lamb'], type=str, help='Optimizer')
    parser.add_argument('--weight_decay_rate', default=1e-4, type=float, help='Weight decay rate')
    parser.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16', 'fp16'], type=str, help='Data type')
    parser.add_argument('--steps_per_loop', default=10, type=int, help='Steps per loop (unavailable for Keras fit in TF 2.2, will be added in later version)')
    parser.add_argument('--validation_freq', default='1', type=str, help='Validation frequency. Run eval after specified epochs. Single values mean run every nth epoch.\
            Else specify epochs with comma separation: E.g.: 5,10,15. Default: Run after every epoch')
    parser.add_argument('--time_history_log_steps', default=10, type=int, help='Frequency with which to log timing information with TimeHistory.')
    add_bool_arg(parser, 'use_tpu', default=True, help='Use TPU')
    add_bool_arg(parser, 'save_model', default=True, help='Save model checkpoint(s)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
