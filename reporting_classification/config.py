# Pretrained models configuration, add model configuration here

PRETRAINED_MODELS = {
        'bert_small_uncased': {
            'hub_url': 'tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
            'config': 'bert_config_small_uncased.json',
            'is_tfhub_model': True,
            'vocab_file': 'bert-small-uncased-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': False
            },
        'bert_medium_uncased': {
            'hub_url': 'tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
            'config': 'bert_config_medium_uncased.json',
            'is_tfhub_model': True,
            'vocab_file': 'bert-medium-uncased-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': False
            },
        'bert_base_uncased': {
            'hub_url': 'google/small_bert/bert_uncased_L-12_H-768_A-12/2',
            'config': 'bert_config_base_uncased.json',
            'is_tfhub_model': True,
            'vocab_file': 'bert-base-uncased-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': False
            },
        'bert_large_uncased': {
            'bucket_location': 'pretrained_models/bert/keras_bert/uncased_L-24_H-1024_A-16',
            'hub_url': 'tensorflow/bert_en_uncased_L-24_H-1024_A-16/2',
            'config': 'bert_config_large_uncased.json',
            'is_tfhub_model': True,
            'vocab_file': 'bert-large-uncased-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': False
            },
        'bert_large_uncased_wwm': {
            'bucket_location': 'pretrained_models/bert/keras_bert/wwm_uncased_L-24_H-1024_A-16',
            'hub_url': 'tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2',
            'config': 'bert_config_large_uncased_wwm.json',
            'is_tfhub_model': True,
            'vocab_file': 'bert-large-uncased-whole-word-masking-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': True
            },
        'covid-twitter-bert': {
            'hub_url': 'digitalepidemiologylab/covid-twitter-bert/2',
            'is_tfhub_model': True,
            'config': 'bert_config_covid_twitter_bert.json',
            'vocab_file': 'bert-large-uncased-whole-word-masking-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': True
            },
        'bert_multi_cased': {
            'bucket_location': 'pretrained_models/bert/keras_bert/multi_cased_L-12_H-768_A-12',
            'hub_url': 'tensorflow/bert_multi_cased_L-12_H-768_A-12/2',
            'config': 'bert_config_multi_cased.json',
            'is_tfhub_model': True,
            'vocab_file': 'bert-multi-cased-vocab.txt',
            'lower_case': False,
            'do_whole_word_masking': False
            }
        }