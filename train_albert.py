from transformers import BertTokenizer, BertJapaneseTokenizer, AlbertTokenizer
from transformers import BertForQuestionAnswering, AlbertForQuestionAnswering
import torch
from utils import *
from data_utils import *
from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import pickle
import re
from collections import OrderedDict
import argparse
# from tensorboardX import SummaryWriter

def build(args):
    TAG = create_tags()
    XLSX_PATH = {'train': 'release/train/ca_data', 'dev': 'release/dev/ca_data', 'test': 'release/test/ca_data'}
    
    PRETRAINED_MODEL_NAME = 'ALINEAR/albert-japanese-v2'
    tokenizer = AlbertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    
    train_data = TrainData(XLSX_PATH['train'], TAG, only_positive=args.only_positive)

    trainset = QADataset(train_data.examples, "train", tokenizer=tokenizer)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    dev_data = TrainData(XLSX_PATH['dev'], TAG, only_positive=args.only_positive)
    
    devset = QADataset(dev_data.examples, "train", tokenizer=tokenizer)
    devloader = DataLoader(devset, batch_size=args.batch_size, collate_fn=collate_fn)

    logger.info(f"[train data] {train_data.summary()}")
    logger.info(f"[dev data] {dev_data.summary()}")
    
    test_data = TestData(XLSX_PATH['dev'], TAG)
    testset = QADataset(test_data.examples, "test", tokenizer=tokenizer)
    testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    model = AlbertForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_NAME)
    model = model.to(args.device)

    if args.load_pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model_path))
    
    return model, trainloader, devloader, testloader, tokenizer


def train(args, model, trainloader, devloader, testloader, tokenizer):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_size = len(trainloader) * args.batch_size
    validation_size = len(devloader) * args.batch_size

    min_loss = 10
    max_score = 0
    iters = 0
#     writer = SummaryWriter(f'{args.plot_dir}/bert-base-japanese_lr_{args.lr}_pre{args.load_pretrained_model}')
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for step, data in enumerate(tqdm(trainloader, desc='Training')):
            iters += 1
            tokens_tensors, segments_tensors, \
            masks_tensors, start_labels, end_labels = [t.to(args.device) for t in data[:-3]]
            optimizer.zero_grad()
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            start_positions=start_labels,
                            end_positions=end_labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            running_loss += loss
#             writer.add_scalar('train_loss', loss, iters)
#         writer.add_scalar('avg_train_loss', running_loss / training_size, epoch + 1)
        
        validation_loss = evaluate(model, devloader, args.device)
#         writer.add_scalar('avg_valid_loss', validation_loss / validation_size, epoch + 1)
        
        if not args.make_prediction:    
            if(validation_loss / validation_size < min_loss):
                min_loss = validation_loss / validation_size
                print("model saved!")
                torch.save(model.state_dict(), '{}/lr_{}_pos{}_pre{}.pt'.format(args.model_dir, args.lr, args.only_positive, args.load_pretrained_model))
                    
        logger.info('[epoch %d] train_loss: %.3f, valid_loss: %.3f' %
              (epoch + 1, running_loss / training_size, validation_loss / validation_size))

        if args.make_prediction:
            predictions = predict(model, testloader, args.device, tokenizer)
            out_path = '{}/lr_{}_pos{}_pre{}_ep{}.csv'.format(args.output_dir, args.lr, args.only_positive, args.load_pretrained_model, epoch+1)
            
#             new_predictions = func('dev', predictions)
            formatter(predictions, out_path = out_path)
            
            scores = score("release/dev/dev_ref.csv", out_path)
#             writer.add_scalar('f1', scores, epoch + 1)
            if scores > max_score:
                max_score = scores
                print("model saved!")
                torch.save(model.state_dict(), '{}/lr_{}_pos{}_pre{}.pt'.format(args.model_dir, args.lr, args.only_positive, args.load_pretrained_model))
            logger.info('[epoch %d] scores: %.6f' % (epoch + 1, scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=10, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size.')
    parser.add_argument('--lr', type=float, default=8e-6, help='learning rate.')
    parser.add_argument('--only_positive', type=int, default=0, help='train on only positive sample')
    parser.add_argument('--pretrained_model_path', type=str, default='', help='pretrain model name')
    parser.add_argument('--load_pretrained_model', type=int, default=0, help='whether to load pretrained model')
    parser.add_argument('--make_prediction', type=int, default=0, help='whether to make predictions')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print("device: ", args.device)
    
    args.model_dir = 'alb_save_0627'
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    args.log_dir = 'alb_train_log_0627'
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    args.output_dir = 'alb_output_0627'
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
#     args.plot_dir = 'jap_log'
#     if not os.path.isdir(args.plot_dir):
#         os.makedirs(args.plot_dir)

    global logger
    if args.load_pretrained_model:
        logger = get_logger(args.log_dir + '/train_log_lr_{}_pre{}.txt'.format(args.lr, args.load_pretrained_model))
    else:
        logger = get_logger(args.log_dir + '/train_log_lr_{}.txt'.format(args.lr))
    
    logger.info(args)
#     logger.info('bert-base-japanese')

    set_seed(args.seed)
    model, trainloader, devloader, testloader, tokenizer = build(args)
    train(args, model, trainloader, devloader, testloader, tokenizer)


if __name__ == '__main__':
    main()

