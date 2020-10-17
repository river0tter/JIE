from utils import *
from data_utils import *
from transformers import BertTokenizer, AlbertTokenizer
from transformers import BertJapaneseTokenizer
from transformers import BertForQuestionAnswering, AlbertForQuestionAnswering
from tqdm import tqdm
import pandas as pd
import os
import pickle
import re
from collections import OrderedDict
import csv
import unicodedata
import argparse

import torch
from torch.utils.data import Dataset, DataLoader


with open('tag_dict.pkl', 'rb') as f:
    TAG = pickle.load(f)

def check_start(target, inputs):
    if(target == ''):
        return inputs
    return ' '+ inputs

def merge(bert_predict, albert_predict):
    final = {}
    for key in albert_predict.keys():
        final[key] = ""
        a_sen = albert_predict[key].split()
        b_sen = bert_predict[key].split()
        bert = {}
        albert = {}
        for s1 in b_sen:
            bert[s1.split(':')[0]] = s1.split(':')[1]
        for s1 in a_sen:
            albert[s1.split(':')[0]] = s1.split(':')[1]

        for k, value in albert.items():
            if(k in bert):
                if len(albert[k]) > len(bert[k]):
                    bert[k] = albert[k]
            else:
                bert[k] = albert[k]

        for k, value in bert.items():
            final[key] += check_start(final[key], "%s:%s"%(k, value))
    return final

def load_and_predict(data_dir, model_type, pretrain_model):
    if model_type == 'bert_japanese':
        model = BertForQuestionAnswering.from_pretrained('cl-tohoku/bert-base-japanese')
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        
    if model_type == 'bert_multilingual':
        model = BertForQuestionAnswering.from_pretrained('bert-base-multilingual-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', tokenize_chinese_chars=False)
        
    if model_type == 'albert':
        model = AlbertForQuestionAnswering.from_pretrained('ALINEAR/albert-japanese-v2')
        tokenizer = AlbertTokenizer.from_pretrained('ALINEAR/albert-japanese-v2')
    
    test_data = TestData(data_dir, TAG)
    testset = QADataset(test_data.examples, "test", tokenizer=tokenizer)
    testloader = DataLoader(testset, batch_size=4, collate_fn=collate_fn)
        
    model = model.to(device)
    model.load_state_dict(torch.load(pretrain_model))
    
    prediction = predict(model, testloader, device, tokenizer)
    prediction = func(data_dir, prediction)
    print('finish loading and predicting from {}!'.format(pretrain_model))
    return prediction #prediction dictionary
    
def main(arg):
    set_seed(123)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_jap = load_and_predict(arg, 'bert_japanese', 'save_model/bert_jap.pt')
    bert_mtl1 = load_and_predict(arg, 'bert_multilingual', 'save_model/bert_mul1.pt')
    bert_mtl2 = load_and_predict(arg, 'bert_multilingual', 'save_model/bert_mul2.pt')
    albert = load_and_predict(arg, 'albert', 'save_model/albert.pt')
    final = merge(bert_jap, bert_mtl1)
    final = merge(final, bert_mtl2)
    final = merge(final, albert)
    formatter(final, out_path='./prediction.csv')
#     scores = score("./release/dev/dev_ref.csv", './prediction.csv')
#     print('score: ', scores)
    
if __name__ == '__main__':
    main(sys.argv[1]) # argv[1] is the test data directory.
