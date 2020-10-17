import os
import sys
import logging
import logging.handlers

import json
import re
import numpy as np
import pandas as pd
import csv
import random
from tqdm import tqdm
import unicodedata
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def evaluate(model, dataloader, device):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='evaluating'):
            data = [t.to(device) for t in data[:-3]]
            tokens_tensors, segments_tensors, masks_tensors, start_labels, end_labels = data[:5]
            loss, start_scores, end_scores = model(input_ids=tokens_tensors, 
                                            token_type_ids=segments_tensors, 
                                            attention_mask=masks_tensors,
                                            start_positions=start_labels,
                                            end_positions=end_labels)
            total_loss += loss.item()
    return total_loss

def predict(model, dataloader, device, tokenizer):
    predictions = None
    correct = 0
    total = 0
    total_loss = 0
    predictions = {}
    iters = 0
    model.eval()
    with torch.no_grad():
        history = {}
        for data in tqdm(dataloader):
            iters += 1
            ids = data[5]
            tables = data[6]
            tags = data[7]
            data = [t.to(device) for t in data[:-3]]
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            
            start_scores, end_scores = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            start_scores = torch.nn.functional.softmax(start_scores)
            end_scores = torch.nn.functional.softmax(end_scores)
            for i in range(len(data[0])):
                id = ids[i]
                table = tables[i]
                tag = tags[i]
                all_tokens = tokenizer.convert_ids_to_tokens(tokens_tensors[i])
                scorer, start, end = find_start_end(start_scores[i], end_scores[i])
                is_answerable = torch.sigmoid(torch.tensor([scorer[0], (start_scores[i][start] + end_scores[i][end])]))
                is_answerable = torch.tensor([scorer[0], (start_scores[i][start] + end_scores[i][end])])
                for idx in table.keys():
                    if ((int(id),idx)) not in predictions:
                        predictions[(int(id),idx)] = ''
                if is_answerable[0] <= is_answerable[1]:
                    ans = find_sentence_index(table, start, end)
                    temp = ''.join(all_tokens[start:end])
                    temp = clean_word(temp).split('[SEP]')
                    for i, idx in enumerate(ans):
                        if(temp[i] != ''):
                            if((id+'-'+str(idx), tag, temp[i]) not in history):
                                if(predictions[(int(id),idx)] != ''):
                                    predictions[(int(id),idx)] += ' '
                                predictions[(int(id),idx)] += tag +':'+ temp[i]
                                history[(id+'-'+str(idx), tag, temp[i])] = idx
    predictions = {k: predictions[k] for k in sorted(predictions.keys())}
                                
    return predictions

# def predict(model, tokenizer, dataloader, device):
#     predictions = None
#     correct = 0
#     total = 0
#     total_loss = 0
#     predictions = {}
#     iters = 0
#     with torch.no_grad():
#         history = {}
#         for data in tqdm(dataloader):
#             iters += 1
#             ids = data[5]
#             tables = data[6]
#             tags = data[7]
#             data = [t.to(device) for t in data[:-3]]
#             tokens_tensors, segments_tensors, masks_tensors = data[:3]
            
#             start_scores, end_scores = model(input_ids=tokens_tensors, 
#                             token_type_ids=segments_tensors, 
#                             attention_mask=masks_tensors)
#             start_scores = torch.nn.functional.softmax(start_scores)
#             end_scores = torch.nn.functional.softmax(end_scores)
#             for i in range(len(data[0])):
#                 id = ids[i]
#                 table = tables[i]
#                 tag = tags[i]
#                 all_tokens = tokenizer.convert_ids_to_tokens(tokens_tensors[i])
# #                 print(all_tokens)
#                 scorer, start, end = find_start_end(start_scores[i], end_scores[i])
#                 is_answerable = (torch.tensor([scorer[0], (start_scores[i][start] + end_scores[i][end])]))
#                 for idx in table.keys():
#                     if ((int(id),idx)) not in predictions:
#                         predictions[(int(id),idx)] = ''
#                 if(is_answerable[0] <= is_answerable[1]*1.25):
#                     ans = find_sentence_index(table, start, end)
#                     temp = ''.join(all_tokens[start:end])
#                     print(temp)
#                     temp = clean_word(temp).split('[SEP]')
#                     for i, idx in enumerate(ans):
#                         if(temp[i] != ''):
#                             if((id+'-'+str(idx), tag, temp[i]) not in history):
#                                 if(predictions[(int(id),idx)] != ''):
#                                     predictions[(int(id),idx)] += ' '
#                                 predictions[(int(id),idx)] += tag +':'+ temp[i]
#                                 history[(id+'-'+str(idx), tag, temp[i])] = idx
#     return predictions



############ processing and evaluating helper functions ################

def formatter(predictions, out_path = 'test.csv'):
    import pandas as pd
    df = pd.DataFrame(predictions.items(), columns=['ID', "Prediction"])
    df.sort_values(by=['ID'])
    df['ID'] = [str(i[0])+'-'+str(i[1]) for i in df['ID']]
    df['Prediction'] = ['NONE' if(i == '') else i for i in df['Prediction']]
    df.to_csv(out_path, index=False)
       
def score(ref_file, pred_file):
    with open(ref_file) as csvfile:
        reader = csv.DictReader(csvfile)
        ref_data = list(reader)

    with open(pred_file) as csvfile:
        reader = csv.DictReader(csvfile)
        pred_data = list(reader)

    f_score = 0.0
    for ref_row, pred_row in zip(ref_data, pred_data):
        refs = set(ref_row["Prediction"].split())
        preds = set(pred_row["Prediction"].split())

        p = len(refs.intersection(preds)) / len(preds) if len(preds) > 0 else 0.0
        r = len(refs.intersection(preds)) / len(refs) if len(refs) > 0 else 0.0
        f = 2*p*r / (p+r) if p + r > 0 else 0
        f_score += f

    return f_score / len(ref_data)

def find_sentence_index(index_table, start, end):
    ans = []
    for idx, point in index_table.items():
        ss, es = point
        if(ss >= end):
            pass
        elif(es <= start):
            pass
        else:
            ans.append(idx)
    return ans

def clean_word(sentence):
    stop_words = ['##', '[UNK]', '[CLS]', '▁']
    for s in stop_words:
        sentence = sentence.replace(s, '')
    if(sentence):
        if(sentence[0]=="《" and sentence[-1] != "》"):
            sentence +=  "》"
        if(sentence[0]!="《" and sentence[-1] == "》"):
            sentence = "《"+sentence
    return sentence

# def find_start_end(start_scores, end_scores):
#     start, end = torch.argmax(start_scores), torch.argmax(end_scores)
#     scorer = start_scores+end_scores
#     retry = 1
#     while(start > end or (not start and end)):
#         retry += 1
#         if(retry > len(start_scores)):
#             return scorer, start, end
#         _, starts = torch.topk(start_scores, retry)
#         _, ends = torch.topk(end_scores, retry)

#         if start == 0:
#             start = starts[retry-1]
#         else:
#             if(start_scores[start]+end_scores[ends[retry-1]] > start_scores[starts[retry-1]]+end_scores[end]):
#                 end = ends[retry-1]
#             else:
#                 start = starts[retry-1]
#     return scorer, start, end

def find_start_end(start_scores, end_scores):
    start, end = torch.argmax(start_scores), torch.argmax(end_scores)
    scorer = start_scores+end_scores
    retry = 1
    while(start > end or start==0 or end==0):
        retry += 1
        if(retry > len(start_scores)):
            return scorer, start, end
        _, starts = torch.topk(start_scores, retry)
        _, ends = torch.topk(end_scores, retry)

        if start == 0:
            start = starts[retry-1]
        else:
            if end == 0:
                end = ends[retry-1]
            elif(start_scores[start]+end_scores[ends[retry-1]] > start_scores[starts[retry-1]]+end_scores[end]):
                end = ends[retry-1]
            else:
                start = starts[retry-1]
    return scorer, start, end

def create_tags():
    TAG = {}
    with open('release/README.md', 'r') as f:
        lines = f.readlines()
        t = []
        for l in lines:
            if len(re.findall(r'\d+', l)) != 0:
                t.append(l)
        for i in t[:20]:
            s = i.split('|')
            n = s[1].replace(' ', '')
            c = s[2].replace(' ', '')
            TAG[n] = c
    return TAG

def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


####################  convert between fullwidth and halfwidth font #################
def full2half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)

def convert(value, text):
    text = text.replace(' ', '')
    new_text = full2half(text)
    s = new_text.find(value)
    if s != -1 and (s+len(value)) <= len(text):
        return text[s:s+len(value)]
    return value

# revise PATH for testing
def func(data_dir, _dict):
    file_id = None
    new_dict = {}
    PATH = '%s/{}.pdf.xlsx' % data_dir
    for (fid, idx), preds in _dict.items():
        if preds == '':
            new_dict[(fid, idx)] = ''
        else:
            if fid != file_id:
                file_id = fid
                df = pd.read_excel(PATH.format(fid))
            text = df['Text'][df['Index'] == int(idx)].item()
            ans = ''
            for pred in preds.split(' '):
                s = pred.find(':')
                tag = pred[:s]
                ans += tag + ':'
                value = pred[s+1:]
                new_value = convert(value, text)
                ans += new_value + ' '
            new_dict[(fid, idx)] = ans[:-1]
    return new_dict

def check(_dict, new_dict):
    for key in _dict.keys():
        if _dict[key] != new_dict[key]:
            print("old: ", _dict[key])
            print("new: ", new_dict[key])
            print()