# Third Party
from fileinput import filename
import librosa
import numpy as np
import torch


# ===============================================
#       code from Arsha for loading data.
# This code extract features for a give audio file
# ===============================================
def load_wav(audio_filepath, sr, min_dur_sec=4):
    audio_data,fs  = librosa.load(audio_filepath,sr=16000)
    len_file = len(audio_data)
    
    if len_file <int(min_dur_sec*sr):
        dummy=np.zeros((1,int(min_dur_sec*sr)-len_file))
        extened_wav = np.concatenate((audio_data,dummy[0]))
    else:
        
        extened_wav = audio_data
    return extened_wav

def lin_mel_from_wav(wav, hop_length, win_length, n_mels):
    linear = librosa.feature.melspectrogram(wav, n_mels=n_mels, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def feature_extraction(filepath,sr=16000, min_dur_sec=4,win_length=400,hop_length=160, n_mels=40, spec_len=400,mode='train'):
    audio_data = load_wav(filepath, sr=sr,min_dur_sec=min_dur_sec)
    linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_fft=512)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)
    return (mag_T - mu) / (std + 1e-5)
    
def load_data(filepath,sr=16000, min_dur_sec=4,win_length=400,hop_length=160, n_mels=40, spec_len=400,mode='train'):
    audio_data = load_wav(filepath, sr=sr,min_dur_sec=min_dur_sec)
    #linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_mels)
    linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_fft=512)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    
    if mode=='train' :
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
        # spec_mag = mag_T
    
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)
    
def load_npy_data(filepath,spec_len=400,mode='train'):
    mag_T = np.load(filepath)
    if mode=='train':
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
        # spec_mag = mag_T
    return spec_mag
    
def speech_collate(batch):
    targets = []
    specs = []
    for sample in batch:
        specs.append(sample['features'])
        targets.append((sample['labels']))
    return specs, targets

def load_key(fileName = "./preGeneratedKey.txt"):
    f = open(fileName)
    key = f.read()
    key = list(map(int,key))
    return key

def printRecord(data, epoch, key_string, model_save_path="", file_name=""):
    out = open(f"log/{file_name}_{data.type}.txt","a+") 
    out2 = open(f"log/{file_name}_{data.type}_Detail.txt","a+") 

    record = f'''
-----------------start {data.type} {epoch}-----------------
Total loss {data.mean_loss} and Accuracy {data.mean_acc} after {epoch} epochs\n
model_save_path = {model_save_path}\n
'''
    record_detail = f'''
-----------------start {data.type} {epoch}-----------------
Total loss {data.mean_loss} and Accuracy {data.mean_acc} after {epoch} epochs\n
model_save_path = {model_save_path}\n
'''

    for i in range(len(data.gts)):
        temp1 = hex(int(''.join(['1' if i >= 0.5 else '0' for i in data.gts[i] ]),2))
        temp2 = hex(int(''.join(['1' if i >= 0.5 else '0' for i in data.preds[i] ]),2))
        record_detail += f"----------------- {i} ------------------\nkeys : {temp1}\npreds: {temp2}\n"
        if(key_string == temp1):
            if(key_string != temp2):
                record +="!"
            else:
                record +="+"
        elif (key_string != temp1):
            if(key_string == temp2):
                record +="%"
            else:
                record +="-"

    record += f"\n-----------------end {data.type} {epoch}-----------------"
    record_detail += f"-----------------end {data.type} {epoch}-----------------"

    print(record)
    print(record, file = out)
    print(record_detail, file = out2)

    out.close()

def saveCSV(data, epoch, file_name=""):
    out = open(f"log/{file_name}_{data.type}.csv","a+") 

    record = f'''-----------------start {data.type} {epoch}-----------------\n'''
    temp1 = []
    temp2 = []
    for i in range(len(data.gts)):
        temp1.append(hex(int(''.join(['1' if i >= 0.5 else '0' for i in data.gts[i] ]),2)))
        temp2.append(hex(int(''.join(['1' if i >= 0.5 else '0' for i in data.preds[i] ]),2)))
    record += ','.join([str(i) for i in range(len(data.gts))])
    record += f'\nkeys epoch:{epoch},'
    record += ','.join(temp1)
    record += f'\npredict epoch : {epoch},'
    record += ','.join(temp2)
    print(record, file = out)

    out.close()

def save_model(model, optimizer, epoch, loss, log_filename):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }, f"save_model/{log_filename}.pth")