import os
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

def tag_normalize(tags, df):
    return [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) for tag in df['Tag'].dropna()]

def normalize_tag(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag


class Example():
    def __init__(
        self,
        tag_text, #str
        value_text, #str
        context_text, #OrderedDict of context index and context e.g. OrderedDict([(1, '入 札 公 告'), (2, '次のとおり一般競争入札に付します。'),...])
        file_id,
    ):
        self.tag_text = tag_text
        self.value_text = value_text
        self.context_text = context_text
        self.file_id = file_id

class TestExample():
    def __init__(
        self,
        tag_text,
        context_text,
        file_id,
    ):
        self.tag_text = tag_text
        self.context_text = context_text
        self.file_id = file_id

class TestData():
    def __init__(self, data_path, TAG):
        self.dfs = []
        self.TAG = TAG
        self.examples = []
        self.load_df(data_path)
        
    def load_df(self, data_path):
        for x in tqdm(os.listdir(data_path)):
            df = pd.read_excel(data_path + '/' + x)
            fid = x.split('.')[0]
            self.split_context(df, fid)

    def split_context(self, df, fid):
        dfc = df.copy()
        PI = dfc['Parent Index'].dropna().unique()
        max_index = df['Index'].values[-1]
        total_index = set(df['Index'].values)
        for t in self.TAG.values():
            all_index = []
            for pi in PI:
                all_index.append(int(pi))
                index = dfc['Index'][dfc['Parent Index'] == pi]
                all_index.extend(index)
                context = {}
                for idx in index:
                    context[int(idx)] = dfc['Text'][dfc['Index'] == idx].item()
                context[int(pi)] = dfc['Text'][dfc['Index'] == pi].item()
                context = OrderedDict(sorted(context.items()))
                self.examples.append(TestExample(t, context, fid))
            some_index = total_index - set(all_index)
            
            for i in some_index:
                context = {}
                context[int(i)] = df['Text'][df['Index'] == i].item()
                context = OrderedDict(sorted(context.items()))
                self.examples.append(TestExample(t, context, fid))

class TrainData():
    def __init__(self, data_path, TAG, only_positive=False):
        self.only_positive = only_positive
        self.dfs = [] #list of dataframe
        self.TAG = TAG
        self.TAG_NAME = set(TAG.values())
        self.examples = [] #list of dataframe
        self.load_df(data_path)
        
    def load_df(self, data_path):
        for x in tqdm(os.listdir(data_path)):
            df = pd.read_excel(data_path + '/' + x)
            fid = x.split('.')[0]
            self.add_positive_examples(df, fid)
            if not self.only_positive:
                new_df = self.create_examples(df)
                self.add_negative_examples(df, new_df, df['Parent Index'].unique(), fid)

    def add_positive_examples(self, df, fid):
        for pair in self.tag_value_context_pair(df):
            tag, value, context = pair
            example = Example(tag_text=tag, value_text=value, context_text=context, file_id=fid)
            self.examples.append(example)
            
    def create_examples(self, df):
        pi_list, i_list, t_list, v_list = [], [], [], []
        samples = []
        for idx, row in df.iterrows():
            if not pd.isna(row['Tag']):
                tags = row['Tag'].split(';')
                values = row['Value'].split(';')
                i = row['Index']
                pi = row['Parent Index']
                if len(tags) == len(values):
                    for t, v in zip(*(tags, values)):
                        pi_list.append(pi)
                        i_list.append(i)
                        t_list.append(normalize_tag(t))
                        v_list.append(v)
                        samples.append((i, normalize_tag(t), v))
                elif len(tags) == 1 and len(values) > len(tags):
                    for v in values:
                        pi_list.append(pi)
                        i_list.append(i)
                        t_list.append(normalize_tag(t))
                        v_list.append(v)
                        samples.append((i, normalize_tag(tags[0]), v))
                elif len(values) == 1 and len(tags) > len(values):
                    for t in tags:
                        pi_list.append(pi)
                        i_list.append(i)
                        t_list.append(normalize_tag(t))
                        v_list.append(v)
                        samples.append((i, normalize_tag(t), values[0]))
        new_df = pd.DataFrame({'PI':pi_list, 'I': i_list, 'T': t_list, 'V': v_list})
        self.dfs.append(new_df)
        return new_df

    def add_negative_examples(self, df, new_df, pi_list, fid):
        for pi in pi_list[1:]: # remove nan
            context = self.find_context_by_pi(df, pi)
            tags = new_df['T'][new_df['PI'] == pi].values
            remain_tags = self.TAG_NAME - set(tags)
            for tag in remain_tags:
                example = Example(tag_text=tag, value_text='', context_text=context, file_id=fid)
                self.examples.append(example)
                
#     def add_negative_examples_sample(self, df, new_df, pi_list, fid):
#         for pi in pi_list: # remove nan
#             context = self.find_context_by_pi(df, pi)
#             tags = new_df['T'][new_df['PI'] == pi].values
#             remain_tags = self.TAG_NAME - set(tags)
#             for tag in remain_tags:
#                 example = Example(tag_text=tag, value_text='', context_text=context, file_id=fid)
#                 self.examples.append(example)

    def find_context(self, df, idx):
        dfc = df.copy()
        parent_i = dfc['Parent Index'][idx]
        index = dfc['Index'][dfc['Parent Index'] == dfc['Parent Index'][idx]]
        context = {}
        for idx in index:
            context[int(idx)] = dfc['Text'][dfc['Index'] == idx].item()
        context[int(parent_i)] = dfc['Text'][dfc['Index'] == parent_i].item()
        context = OrderedDict(sorted(context.items()))
        return context
    
    def find_context_by_pi(self, df, pi):
        dfc = df.copy()
        index = dfc['Index'][dfc['Parent Index'] == pi]
        context = {}
        for idx in index:
            context[int(idx)] = dfc['Text'][dfc['Index'] == idx].item()
        context[int(pi)] = dfc['Text'][dfc['Index'] == pi].item()
        context = OrderedDict(sorted(context.items()))
        return context

    def tag_value_context_pair(self, df):
        pair = []
        for idx, tags, values in zip(*(df['Tag'].dropna().index, tag_normalize(df['Tag'].dropna(), df), df['Value'].dropna())):
            tags = tags.split(';')
            values = values.split(';')
            context = self.find_context(df, idx)
            if len(tags) == len(values):
                for t, v in zip(*(tags, values)):
                    pair.append((t, v, context))
            elif len(tags) == 1 and len(values) > len(tags):
                for v in values:
                    pair.append((tags[0], v, context))
            elif len(values) == 1 and len(tags) > len(values):
                for t in tags:
                    pair.append((t, values[0], context))            
        return pair

    def summary(self):
        cnt = 0
        for e in self.examples:
            if e.value_text == '':
                cnt += 1
        return (f'negative samples / positive samples : {cnt} / {len(self.examples) - cnt}')
    
    
#####################  DATASET ##############################

class QADataset(Dataset):
    def __init__(self, example, mode, tokenizer, MAX_LENGTH=512):
        self.example = example
        self.mode = mode
        self.len = len(example)
        self.tokenizer = tokenizer
        self.max_len = MAX_LENGTH
    
    # 重新計算ans token在context tokens 的位置
    def token_start_end(self, tokenizedTokens, target):
        if not target:
            return 0,0
        target_len = len(target)
        for i, t in enumerate(tokenizedTokens):
            if t == target[0] and i <= len(tokenizedTokens)-target_len:
                if(target == tokenizedTokens[i:i+target_len]):
                    return (i, i+target_len)
        return 0,0
    
    def __getitem__(self, idx):
        ans_token = ""
        if self.mode == "test":
            # contexts = '[SEP]'.join(self.example[idx].context_text.values())
            contexts = self.example[idx].context_text.values()
            indexs = list(self.example[idx].context_text.keys())
            id = self.example[idx].file_id
            tag = self.example[idx].tag_text
        else:
            # contexts = '[SEP]'.join(self.example[idx].context_text.values())
            contexts = self.example[idx].context_text.values()
            indexs = list(self.example[idx].context_text.keys())
            id = self.example[idx].file_id
            tag = self.example[idx].tag_text
            ans_token = self.tokenizer.tokenize(self.example[idx].value_text)
        
        # contexts = contexts[:self.max_len-3-len(tag)]

        index_table = {}
        word_pieces = ["[CLS]"]
        spoint = 1
        for i, text in enumerate(contexts):
            tokens_a = self.tokenizer.tokenize(text)
            index_table[indexs[i]] = (spoint, spoint +len(tokens_a))
            word_pieces += tokens_a + ["[SEP]"]
            spoint += 1+len(tokens_a)
        
        len_a = len(word_pieces)
        if(len_a > self.max_len-3-len(tag)):
            word_pieces = word_pieces[:self.max_len-3-len(tag)]
            if(word_pieces[-1] != '[SEP]'):
                word_pieces += ['[SEP]']
            len_a = len(word_pieces)
        # 找到anwer 在 contexts 中的 start end
        start, end = self.token_start_end(word_pieces, ans_token)
        start_label = torch.tensor(start)
        end_label = torch.tensor(end)

        tokens_b = self.tokenizer.tokenize(tag)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # convert token to indexs
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, start_label, end_label, id, index_table, tag)
    
    def __len__(self):
        return self.len
    
def collate_fn(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    start_label = torch.tensor([s[2] for s in samples])
    end_label = torch.tensor([s[3] for s in samples])
    questions_ids = [s[4] for s in samples]
    table = [s[5] for s in samples]
    tag = [s[6] for s in samples]
    
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, start_label, end_label, questions_ids, table, tag


