"""
This script reuses the model submitted by Valdes et al. (2021) to #SMM4H 2021 for the completion of Task 6 (Classification of COVID-19 tweets containing symptoms).
Valdes et al. (2021) paper was published here:
Proceedings of the Sixth Social Media Mining for Health (#SMM4H) Workshop and Shared Task; Magge, A., Klein, A., Miranda-Escalada, A., Al-garadi, M. A., Alimova, I., Miftahutdinov, Z., Farre-Maduell, E., Lopez, S. L., Flores, I., O’Connor, K., Weissenbacher, D., Tutubalina, E., Sarker, A., Banda, J. M., Krallinger, M., Gonzalez-Hernandez, G., Eds.; Association for Computational Linguistics: Mexico City, Mexico, 2021.
The title of Valdes et al. (2021) paper is: 
    UACH at SMM4H: a BERT based approach for classification of COVID-19 Twitter posts
The full paper can be found as of p. 65 of the Proceedings.

The authors kindly shared their code with us: https://drive.google.com/drive/folders/1R8gcZ0iztTuDRS3Z7MLpJbcNEwl5ENq_?usp=sharing 
The Jupyter notebook hosted on Google Colab contains the implementation of the model with PyTorch.

"""

import os
import json
import pandas as pd
import transformers
import torch
from onnx2keras import onnx_to_keras

class DeepBERT(torch.nn.Module):
    def __init__(self):
        super(DeepBERT, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(1024, 3)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1[1])
        output_3 = self.l3(output_2)
        return output_3

def get_input_data(preprocessed_tweet, tokenizer, max_seq_length, device):
    features_dict = tokenizer(preprocessed_tweet, is_split_into_words=False, padding=True, max_length=max_seq_length)
    
    input_ids = torch.tensor([features_dict['input_ids']]).to(device, dtype=torch.long)
    attention_mask = torch.tensor([features_dict['attention_mask']]).to(device, dtype=torch.long)
    token_type_ids = torch.tensor([features_dict['attention_mask']]).to(device, dtype=torch.long)

    return torch.autograd.Variable(input_ids), torch.autograd.Variable(attention_mask), torch.autograd.Variable(token_type_ids)

def main():
    model_dir = 'selfreporting_model_finetuned'
    model = DeepBERT()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()). 
    # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. ct_bert_task6_params_dict is such a state_dict.

    # Load model weights (the file ct_bert_task6_params_dict can be downloaded via the link to the Google Drive folder above)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'ct_bert_task6_params_dict'), map_location='cuda:0'))
    model.eval()

    # Create a file that defines both the architecture and the weights of the model
    torch.save(model, os.path.join(model_dir, 'pytorch_model.bin'))


if __name__=='__main__':
    main()
