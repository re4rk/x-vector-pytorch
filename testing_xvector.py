#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""



import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath',type=str,default='meta/training.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing.txt')
parser.add_argument('-validation_filepath',type=str, default='meta/validation.txt')

parser.add_argument('-input_dim', action="store_true", default=257)
parser.add_argument('-num_classes', action="store_true", default=2)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=16)
parser.add_argument('-use_gpu', action="store_true", default=True)
args = parser.parse_args()

### Data related
dataset_test = SpeechDataGenerator(manifest=args.testing_filepath,mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"device is {device}")
model = X_vector(args.input_dim, args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()




def test(dataloader_val):
    print("start test")
    checkpoint = torch.load("save_model/best_check_point_99_0.000825372063748849")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        test_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]])).long()
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #### CE loss
            loss = loss_fun(pred_logits,labels)
            test_loss_list.append(loss.item())
            if i_batch%10==0:
                print('Loss {} after {} iteration'.format(np.mean(np.asarray(test_loss_list)),i_batch))
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(test_loss_list))
        print(f'Total test loss {mean_loss} and Test accuracy {mean_acc}')
        print(full_preds[:80])
        print(full_gts[:80])

        model_save_path = os.path.join('save_test', 'best_check_point_'+str(mean_loss))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': checkpoint['epoch']}
        torch.save(state_dict, model_save_path)


if __name__ == '__main__':
    test(dataloader_test)