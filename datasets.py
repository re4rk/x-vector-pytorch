#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:09:44 2020

@author: krishna
"""

import os
import numpy as np
import glob
import argparse

# get folders in the folder
def get_folders(path):
    folders = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            folders.append(name)
    return folders

class_ids ={'man':0,'other':1}

def save_meta_file(file_path, meta_file_path):
    fid = open(meta_file_path,'w')
    for filepath in file_path:
        fid.write(filepath+'\n')
    fid.close()

def create_meta(files_list, store_loc, mode='train', user_id='id10300'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
    
    # make user specific folders in the meta folder
    user_loc = os.path.join(store_loc, user_id)
    if not os.path.exists(user_loc):
        os.makedirs(user_loc)

    try:
        if mode=='train':
            meta_store = os.path.join(user_loc, 'training.txt')
        elif mode=='test':
            meta_store = os.path.join(user_loc, 'testing.txt')
        elif mode=='validation':
            meta_store = os.path.join(user_loc, 'validation.txt')
        save_meta_file(files_list, meta_store)
    except Exception as e:
        print('Error in creating meta files')
        print(e)

def save_selected_file(nums, sub_files, label, train_lists):
    for i in nums:
        to_write = sub_files[i]+' '+str(class_ids[label])
        train_lists.append(to_write)
    return train_lists

def extract_files(folder_path, user_id):
    all_lang_folders = sorted(glob.glob(folder_path+'/*/'))
    train_lists=[]
    test_lists = []
    val_lists=[]
    
    for lang_folderpath in all_lang_folders:
        if user_id in lang_folderpath:
            language = "man"
        else :
            language = "other"

        sub_folders = glob.glob(lang_folderpath+'/*/')

        files = []
        for sub_folder in sub_folders:
            all_files = glob.glob(sub_folder+'/*.wav')
            files.extend(all_files)
        
        np.random.shuffle(files)

        zero_to_train = len(sub_folders) - int(len(sub_folders)*0.4)
        train_nums = range(zero_to_train)
        train_lists = save_selected_file(train_nums, files, language, train_lists)

        train_to_val = len(sub_folders) - int(len(sub_folders)*0.2)
        val_nums = range(zero_to_train, train_to_val)
        val_lists = save_selected_file(val_nums, files, language, val_lists)

        val_to_test = len(sub_folders)
        test_nums = range(train_to_val, val_to_test)
        test_lists = save_selected_file(test_nums, files, language, test_lists)
    return train_lists, test_lists, val_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--processed_data", default="../vox1_test_wav", type=str,help='Dataset path')
    parser.add_argument("--meta_store_path", default="meta/", type=str,help='Save directory after processing')
    config = parser.parse_args()

    users = get_folders(config.processed_data)
    print("users : ", end="")
    print(users)
    for user in users:
        train_list, test_list,val_lists = extract_files(config.processed_data,user_id=user)

        create_meta(train_list,config.meta_store_path,mode='train',user_id=user)
        create_meta(test_list,config.meta_store_path,mode='test',user_id=user)
        create_meta(val_lists,config.meta_store_path,mode='validation',user_id=user)