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
parser.add_argument('-evaling_filepath',type=str, default='meta/testing.txt')

parser.add_argument('-input_dim', action="store_true", default=257)
parser.add_argument('-num_classes', action="store_true", default=2)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=1)
parser.add_argument('-use_gpu', action="store_true", default=True)
args = parser.parse_args()

### Data related
dataset_eval = SpeechDataGenerator(manifest=args.evaling_filepath,mode='eval')
dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"device is {device}")
model = X_vector(args.input_dim, args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()




def eval(dataloader_eval):
    print("start eval")
    checkpoint = torch.load("save_model/best_check_point_99_0.000825372063748849")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        eval_loss_list=[]
        full_preds=[]
        full_gts=[]
        for sample_batched in dataloader_eval:
            features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]])).long()
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #### CE loss
            loss = loss_fun(pred_logits,labels)
            eval_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            
            full_preds.extend(predictions)
            full_gts.extend(labels.detach().cpu().numpy().tolist())
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(eval_loss_list))
        print(f'Total eval loss {mean_loss} and eval accuracy {mean_acc}')
        print(full_preds)
        print(full_gts)

        model_save_path = os.path.join('save_eval', 'best_check_point_'+str(mean_loss))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': checkpoint['epoch']}
        torch.save(state_dict, model_save_path)


if __name__ == '__main__':
    eval(dataloader_eval)