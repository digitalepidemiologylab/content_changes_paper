USAGE_DESCRIPTION = """
Run prediction by loading a fine-tuned model
"""

import datetime
import time
import math
from tqdm import tqdm

import argparse
import os
import json

import sys
# Import from official repo
sys.path.append('tensorflow_models')
from official.utils.misc import distribution_utils

from input_pipeline_multinomial import single_file_dataset
from utils.misc import ArgParseDefault, add_bool_arg, save_to_json
import utils.optimizer

import collections

import pandas as pd
import tensorflow as tf
# Remove duplicate logger (not sure why this is happening, possibly an issue with the imports)
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

# Add file logging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

VOCAB_PATH = 'vocabs'

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

def read_run_log(run_dir):
    with tf.io.gfile.GFile(os.path.join(run_dir, 'run_logs.json'), 'rb') as reader:
        run_log = json.loads(reader.read().decode('utf-8'))
    return run_log

def get_input_meta_data(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, 'meta.json'), 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))
    return input_meta_data

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

def format_prediction(preds, label_mapping, label_name):
    preds = tf.nn.softmax(preds, axis=1)
    formatted_preds = []
    for pred in preds.numpy():
        # Convert to Python types
        pred = {label: float(probability) for label, probability in zip(label_mapping.values(), pred)}
        # Sort by probabilities (i.e. item[1]) in decreasing order
        pred = {k: v for k, v in sorted(pred.items(), key=lambda item: item[1], reverse=True)}
        # The predicted label "value" corresponds to the first key in the above dictionary
        formatted_preds.append({label_name: list(pred.keys())[0], f'{label_name}_probabilities': pred})
    return formatted_preds

def create_tfrecord_dataset_pipeline(input_file, max_seq_length, batch_size, input_pipeline_context=None):
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64)
    }
    dataset = single_file_dataset(input_file, name_to_features)

    # Shard dataset between hosts
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines, input_pipeline_context.input_pipeline_id)

    def _select_data_from_record(record):
        x = {
            'input_ids': record['input_ids'],
            'token_type_ids': record['segment_ids'],
            'attention_mask': record['input_mask']
            }
        return x

    dataset = dataset.map(_select_data_from_record)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(1024)
    return dataset

def get_tfrecord_dataset(input_file, eval_batch_size, max_seq_length):
    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset for distributed prediction."""
        batch_size = ctx.get_per_replica_batch_size(global_batch_size) if ctx else eval_batch_size
        dataset = create_tfrecord_dataset_pipeline(input_file, max_seq_length, batch_size, input_pipeline_context=ctx)
        return dataset
    return _dataset_fn

def run(args):
    # Start time
    s_time = time.time()
    
    # Input data
    # A) TFRecord files
    input_dir = f'gs://{args.bucket_name}/data/predictions_multinomial/unlabelled_data/{args.input_folder}'
    tfrecord_folder = os.path.join(input_dir, 'tfrecords')
    tfrecord_filenames_list = tf.io.gfile.listdir(tfrecord_folder)
    tfrecord_filepaths_list = [os.path.join(tfrecord_folder, element) for element in tfrecord_filenames_list]
    
    # B) Read run log
    run_dir = f'gs://{args.bucket_name}/data/finetune_multinomial/runs/{args.run_name}'
    run_log = read_run_log(run_dir)
    
    # C) Read JSON files generated when training the model
    training_dir = f'gs://{args.bucket_name}/data/finetune_multinomial/finetune_data/{run_log["finetune_data"]}'
    # input_meta_data = get_input_meta_data(training_dir)
    
    # Creating output folder
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    output_folder = os.path.join('data', 'predictions_multinomial', f'predictions_{ts}')
    output_folder = f'gs://{args.bucket_name}/data/predictions_multinomial/predictions_{ts}'
    predictions_output_folder = os.path.join(output_folder, 'predictions')
    if not tf.io.gfile.isdir(predictions_output_folder):
        tf.io.gfile.makedirs(predictions_output_folder)

    # Characteristics given in run_log
    max_seq_length = run_log['max_seq_length']
    label_mapping = run_log['label_mapping']
    num_labels = len(label_mapping)
    num_epochs = run_log['num_epochs']
    warmup_proportion = run_log['warmup_proportion']

    # Arithmetic operations leading to warmup_steps
    # Characteristics of the training set
    train_batch_size = run_log['train_batch_size']
    train_data_size = len(run_log['all_predictions'][0])
    # Arguments to pass to utils.optimizer.create_optimizer
    steps_per_epoch = int(math.floor(train_data_size / train_batch_size))
    warmup_steps = int(num_epochs * train_data_size * warmup_proportion / train_batch_size)
    learning_rate = run_log['learning_rate']
    end_lr = run_log['end_lr']
    optimizer_type = run_log['optimizer_type']

    # Load model
    logger.info(f'Loading model...')
    # Create optimizer
    optimizer = utils.optimizer.create_optimizer(learning_rate,
                                                steps_per_epoch,
                                                warmup_steps,
                                                end_lr,
                                                optimizer_type)
    optimizer = configure_optimizer(optimizer, use_float16=False, use_graph_rewrite=False)
    loss_fn = get_loss_fn(num_labels)
    model_path = os.path.join(run_dir, 'model')
    logger.info(f'Restoring model from {model_path}...')
    model = tf.keras.models.load_model(model_path, custom_objects={'AdamWeightDecay': optimizer, 'classification_loss_fn': loss_fn})
    logger.info(f'...Restored model from {model_path}')

    # Batch size for the unlabelled data
    test_batch_size = args.test_batch_size
    # Predict
    num_predictions = 0
    predictions = []

    s_time_predict = time.time()
    for input_filepath in tfrecord_filepaths_list:
        f_out_filename = os.path.basename(input_filepath).split('.')[-2]
        dataset = get_tfrecord_dataset(input_filepath, test_batch_size, max_seq_length)()
        batch_number = 0
        num_batches = sum(1 for _ in tf.data.TFRecordDataset(input_filepath).batch(test_batch_size))
        for batch in tqdm(dataset, total=num_batches, unit='batch'):
            batch_number += 1
            k = batch_number // args.batch_output_split
            f_out_path = os.path.join(predictions_output_folder, f'{f_out_filename}_{k}.jsonl')
            preds = model.predict(batch, steps=None)
            preds = format_prediction(preds, label_mapping, args.label_name)
            num_predictions += len(preds)
            with tf.io.gfile.GFile(f_out_path, 'a') as f:
                for pred in preds:
                    f.write(json.dumps(pred) + '\n')
    e_time_predict = time.time()
    prediction_time_min = (e_time_predict - s_time_predict)/60
    logger.info(f'Wrote {num_predictions:,} predictions in {prediction_time_min:.1f} min ({num_predictions/prediction_time_min:.1f} predictions per min)')

    # End time
    e_time = time.time()
    total_time_min = (e_time - s_time)/60
    
    # Write config 
    f_config = os.path.join(output_folder, 'predict_config.json')
    logger.info(f'Saving config to {f_config}')
    data = {
            'prediction_time_min': prediction_time_min,
            'total_time_min': total_time_min,
            'num_predictions': num_predictions,
            **vars(args)}
    save_to_json(data, f_config)

def main(args):
    # Set TF Hub caching to bucket
    os.environ['TFHUB_CACHE_DIR'] = os.path.join(f'gs://{args.bucket_name}/tmp')
    # Get distribution strategy
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
    # Run training
    with strategy.scope():
        run(args)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault(usage=USAGE_DESCRIPTION)
    parser.add_argument('--bucket_name', required=True, help='Bucket name')
    parser.add_argument('--test_batch_size', default=1024, type=int, help='The batch size of any model should always be at least 64 (8 per TPU core), since the TPU always pads the tensors to this size. The ideal batch size when training on the TPU is 1024 (128 per TPU core), since this eliminates inefficiencies related to memory transfer and padding. It is recommended to use the largest batch size which fits in to memory and is a multiple of 64. The easiest way to achieve this is to start with 1024, and if this causes an out-of-memory error then try reducing the batch size until the model runs successfully. ')
    parser.add_argument('--batch_output_split', required=True, type=int, help='Number of batches reported per output JSON Line file.')
    parser.add_argument('--label_name', default='label', type=str, help='Assign a name to the classification problem')
    parser.add_argument('--input_folder', required=True, help='Folder containing the TFRecord file(s)')
    parser.add_argument('--run_name', required=True, help='Finetune run name. The model will be loaded from gs://{bucket_name}/data/finetune_multinomial/runs/{run_name}.')
    parser.add_argument('--tpu_name', required=False, help='Name of the TPU (required for pods)')
    parser.add_argument('--tpu_ip', required=False, help='IP-address of the TPU')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    add_bool_arg(parser, 'use_tpu', default=True, help='Use TPU')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
