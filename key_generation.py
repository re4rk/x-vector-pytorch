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

class Args:
    training_filepath = 'meta/training.txt'
    testing_filepath = 'meta/testing.txt'
    validation_filepath = 'meta/validation.txt'
    input_dim = 257
    num_classes = 2
    lamda_val = 0.1
    batch_size = 1
    use_gpu = True
    num_epochs = 100
    mode = "test"
args=Args()

### Data related
dataloader_eval = SpeechDataGenerator(manifest=args.testing_filepath,mode='test')
dataloader_eval = DataLoader(dataloader_eval, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"device is {device}")
model = X_vector(args.input_dim, args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()

print("start eval")
checkpoint = torch.load("save_model/best_check_point_99_0.000825372063748849")
model.load_state_dict(checkpoint['model'])
model.eval()
with torch.no_grad():
    eval_loss_list=[]
    full_preds_before=[]
    full_preds=[]
    full_gts=[]
    full_x_vec=[]
    for sample_batched in dataloader_eval:
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]])).long()
        features, labels = features.to(device),labels.to(device)
        pred_logits,x_vec = model(features)
        full_x_vec.append(x_vec)
        #### CE loss
        loss = loss_fun(pred_logits,labels)
        eval_loss_list.append(loss.item())
        #train_acc_list.append(accuracy)
        full_preds_before.extend(pred_logits)
        prediction = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        
        full_preds.append(np.int64(prediction[0]))
        full_gts.append(np.int64(labels.detach().cpu().numpy()[0]))
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(eval_loss_list))
    print(f'Total eval loss {mean_loss} and eval accuracy {mean_acc}')
    print(full_preds_before[:2])
    print(''.join(map(str,full_preds)))
    print(''.join(map(str,full_gts)))

    model_save_path = os.path.join('save_eval', 'best_check_point_'+str(mean_loss))
    state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': checkpoint['epoch']}
    torch.save(state_dict, model_save_path)

for idx, x_vec in enumerate(full_x_vec):
    key_len = 64
    b = F.normalize(x_vec,dim=0)
    print(b)
    b = torch.split(x_vec,[512//key_len]*(key_len),dim=1)
    b = torch.stack(list(b), dim=1)
    b = torch.sum(b,dim=2)
    generated = ""
    for i in b[0]:
        if(i >= 0):
            generated += "1"
        else :
            generated += "0"
    # if(full_gts[idx] == 1):
    #     print(f"label = {full_gts[idx]}, pred = {full_preds[idx]} ")
    #     print(generated, hex(int(generated, 2)))
        # if(idx == 10):
        #     break
    if(full_gts[idx] == 0):
        print(f"label = {full_gts[idx]}, pred = {full_preds[idx]} ")
        print(generated, hex(int(generated, 2)))
        # if(idx == 10):
        #     break