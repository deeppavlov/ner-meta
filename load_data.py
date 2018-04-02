import os
import json
from glob import glob
import ftplib
import numpy as np
import random
import urllib.request
import tarfile
import re
import pandas as pd


SNIPS_URL = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/ner/SNIPS2017.tar.gz'


def tokenize(s):
    return re.findall(r"[\w']+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)


def download_and_extract_archive(url, extract_path):
    archive_filename = url.split('/')[-1]
    archive_path = os.path.join(extract_path, archive_filename)
    os.makedirs(extract_path, exist_ok=True)
    urllib.request.urlretrieve(SNIPS_URL, archive_path)
    f = tarfile.open(archive_path)
    f.extractall(extract_path)
    f.close()
    os.remove(archive_path)

#download_and_extract_archive(SNIPS_URL, 'data/')

def parse_snips_utterance(utterance):
    if 'data' in utterance:
        utterance_tokens = list()
        utterance_tags = list()
        instances = []
        for item in utterance['data']:
            print(item)
            text = item['text']
            tokens = tokenize(text)
            if 'entity' in item:
                entity = item['entity']
                tags = list()
                for n in range(len(tokens)):
                    if n == 0:
                        tags.append('B-' + entity)
                    else:
                        tags.append('I-' + entity)
            else:
                tags = ['O' for _ in range(len(tokens))]
            instances += [[to, 'O', 'O', ta] for ta, to in zip(tags, tokens)]
            utterance_tags.extend(tags)
            utterance_tokens.extend(tokens)
    print(instances)
    1/0
    return utterance_tokens, utterance_tags

def snips_reader(dataset_download_path='data/', return_intent=False):
    # param: dataset_download_path - path to the existing dataset or if there is no
    #   dataset there the dataset it will be downloaded to this path
    if not os.path.isdir(os.path.join(dataset_download_path, 'AddToPlaylist')):
        download_and_extract_archive(SNIPS_URL, dataset_download_path)
    contents = glob(os.path.join(dataset_download_path, '*'))
    total_train = dict()

    for folder in contents:
        if os.path.isdir(folder):
            folder_name = folder.split('/')[-1]
            train_file_name = 'train_' + folder_name + '_full.json'
            with open(os.path.join(folder, train_file_name), encoding='cp1251') as f:
                total_train.update(json.load(f))
    intetns = list()
    xy_list = list()
    #i=0
    for n, key in enumerate(total_train):
        data = total_train[key]
        for item in data:
            xy_list.append(parse_snips_utterance(item))
            # if i==0:
            #     print(parse_snips_utterance(item))
            #     i+=1
            intetns.append(key)
    if return_intent:
        return xy_list, intetns
    else:
        return xy_list

#print(type(snips_reader()))
print(snips_reader()[0])
#print(snips_reader()[-1])
#tags = np.concatenate([w[1] for w in snips_reader()])
#print(np.unique(tags))