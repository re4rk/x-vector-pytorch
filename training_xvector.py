#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""

import torch
from torch.utils.data import DataLoader
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import numpy as np
from torch import optim
import argparse
from models.x_vector_key4 import X_vector
from utils.utils import speech_collate, load_key, printRecord, saveCSV, save_model
from utils.DataClass import DataClass
import random
import time
import json

with open('config.json') as f:
    config = json.load(f)

torch.multiprocessing.set_sharing_strategy('file_system')

# Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath', type=str, default=config['path']['train'])
parser.add_argument('-testing_filepath', type=str, default=config['path']['test'])
parser.add_argument('-validation_filepath', type=str, default=config['path']['val'])

parser.add_argument('-input_dim', action="store_true", default=config['model']['input_dim'])
parser.add_argument('-num_classes', action="store_true", default=config['model']['num_classes'])
parser.add_argument('-lamda_val', action="store_true", default=config['model']['lamda_val'])
parser.add_argument('-batch_size', action="store_true", default=config['model']['batch_size'])
parser.add_argument('-use_gpu', action="store_true", default=config['model']['use_gpu'])
parser.add_argument('-num_epochs', action="store_true", default=config['model']['num_epochs'])
args = parser.parse_args()

# Data related
dataset_train = SpeechDataGenerator(manifest=args.training_filepath, mode='train')
dataloader_train = DataLoader(dataset_train, num_workers=40, batch_size=args.batch_size, shuffle=False,
                              collate_fn=speech_collate)

dataset_val = SpeechDataGenerator(manifest=args.validation_filepath, mode='validation')
dataloader_val = DataLoader(dataset_val, num_workers=16, batch_size=args.batch_size, shuffle=False,
                            collate_fn=speech_collate)

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath, mode='test')
dataloader_test = DataLoader(dataset_test, num_workers=16, batch_size=args.batch_size, shuffle=False,
                             collate_fn=speech_collate)

# Model related
if args.use_gpu:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device is {device}")
    model = X_vector(args.input_dim, args.num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
else:
    device = torch.device("cpu")
    model = X_vector(args.input_dim, args.num_classes).to(device)
    print("Let's use CPU!")

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.MSELoss()

key = load_key()
key_string = hex(int(''.join(['1' if i >= 0.5 else '0' for i in key]), 2))
other = [0.5 for i in key]

preBatch = []
for sample_batched in dataloader_train:
    preBatch.append(sample_batched)

now = time.localtime()
log_filename = f"{now.tm_year:4}_{now.tm_mon:2}_{now.tm_mday:2}_{now.tm_hour:2}_{now.tm_min:2}".replace(" ", "0")


def key_batch(sample):
    applied_key = []
    for torch_tensor in sample:
        if torch_tensor == 0:
            applied_key.append(key)
        else:
            others = [random.randint(0, 1) for _ in key]
            applied_key.append(others)
    return applied_key


def make_feature(sample):
    array = np.asarray([tensor.numpy().T for tensor in sample])
    return torch.from_numpy(array).float().to(device)


def make_label(label):
    array = np.asarray(key_batch(label))
    return torch.from_numpy(array).float().to(device)


def train():
    data = DataClass("train")
    model.train()
    for i_batch, sample in enumerate(preBatch):
        features, labels = make_feature(sample[0]), make_label(sample[1])

        # requires_grad : 기울기 저장
        features.requires_grad = True
        optimizer.zero_grad()
        pred, x_vec = model(features)

        # CE loss
        loss = loss_fun(pred, labels)
        loss.backward()
        optimizer.step()

        data.add_log(loss, pred, labels)

    data.calculate_mean_acc()
    data.calculate_mean_loss()

    # save log
    if epoch % 10 == 0:
        save_model(model, optimizer, epoch, data.mean_loss, log_filename)

    return data


def validation(dataloader):
    data = DataClass("test")
    model.eval()
    with torch.no_grad():
        for i_batch, sample in enumerate(dataloader):
            features, labels = make_feature(sample[0]), make_label(sample[1])

            # requires_grad : 기울기 저장
            pred, x_vec = model(features)

            # CE loss
            loss = loss_fun(pred, labels)

            data.add_log(loss, pred, labels)

    data.calculate_mean_acc()
    data.calculate_mean_loss()

    return data


if __name__ == '__main__':
    for epoch in range(args.num_epochs):
        train_data = train()
        printRecord(train_data, epoch, key_string, log_filename, log_filename)
        val_data = validation(dataloader_val)
        printRecord(val_data, epoch, key_string, log_filename, log_filename)
        saveCSV(val_data, epoch, log_filename)
    print("hello")
