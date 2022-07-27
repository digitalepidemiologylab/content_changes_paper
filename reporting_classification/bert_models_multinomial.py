# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT models that are compatible with TF 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from transformers import TFBertModel

def classifier_model(bert_config, 
                     model_path,
                     num_labels,
                     max_seq_length,
                     final_layer_initializer=None):
    """BERT classifier model in functional API style.

    Construct a Keras model for predicting `num_labels` outputs from an input with
    maximum sequence length `max_seq_length`.

    Args:
    bert_config: BertConfig or AlbertConfig, the config defines the core BERT or
        ALBERT model.
    model_path: string, the path or url to a PyTorch state_dict save file (e.g, ./pt_model/pytorch_model.bin). 
        In this case, from_pt should be set to True 
        and a configuration object should be provided as config argument.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    final_layer_initializer: Initializer for final dense layer. Defaulted
        TruncatedNormal initializer.

    Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
    """
    if final_layer_initializer is not None:
        initializer = final_layer_initializer
    else:
        initializer = tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range)
    
    encoder = TFBertModel.from_pretrained(model_path)    
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)

    # pooler_output corresponds to the representation of the [CLS] token from the top-most layer. 
    # It's pooling in the sense that it's extracting a representation for the whole sequence. 
    pooler_output = encoder.bert({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})['pooler_output'] 
    output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(pooler_output)
    output = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer, name='output')(output)

    return tf.keras.Model(
      inputs={
          'input_ids': input_ids,
          'token_type_ids': token_type_ids,
          'attention_mask': attention_mask
      },
      outputs=output), encoder
