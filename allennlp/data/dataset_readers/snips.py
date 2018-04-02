from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    line = line.strip()
    return not line or line == """-DOCSTART- -X- -X- O"""


_VALID_LABELS = {'ner', 'pos', 'chunk'}

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


def parse_snips_utterance(utterance):
    if 'data' in utterance:
        # utterance_tokens = list()
        # utterance_tags = list()
        instances = []
        for item in utterance['data']:
            # print(item)
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
            instances += [[to, ta] for ta, to in zip(tags, tokens)]
            # utterance_tags.extend(tags)
            # utterance_tokens.extend(tokens)
    return instances


def snips_reader(file='train', dataset_download_path='data/', return_intent=False):
    # param: dataset_download_path - path to the existing dataset or if there is no
    #   dataset there the dataset it will be downloaded to this path
    if not os.path.isdir(os.path.join(dataset_download_path, 'AddToPlaylist')):
        download_and_extract_archive(SNIPS_URL, dataset_download_path)
    contents = glob(os.path.join(dataset_download_path, '*'))
    total_trains = [dict() for _ in range(7)]
    i = 0
    fs = []
    for folder in contents:
        if os.path.isdir(folder):
            fs.append(folder)
    folders = sorted(fs)
    for folder in folders:
        folder_name = folder.split('/')[-1]
        train_file_name = 'train_' + folder_name + '_full.json'
        with open(os.path.join(folder, train_file_name), encoding='cp1251') as f:
            total_trains[i].update(json.load(f))
            i += 1
    intetns = [list() for _ in range(7)]
    xy_lists = [list() for _ in range(7)]
    i = 0
    for total_train in total_trains:
        for n, key in enumerate(total_train):
            data = total_train[key]
            for item in data:
                xy_lists[i].append(parse_snips_utterance(item))
                intetns[i].append(key)
        i += 1

    n_tasks = 200
    tasks_batch = 30
    task_size = 100

    # if file == 'train':
    #     n_tasks = 20
    #     task_size = 30
    # else:
    #     n_tasks = 1
    #     task_size = 500

    # domains = np.arange(7, dtype=np.int32)
    #
    # for domain in domains:
    #     xy_lists.append([])
    #     this_train = total_trains[domain]
    #     ys = []
    #     for n, key in enumerate(this_train):
    #         data = this_train[key]
    #         for item in data:
    #             ys += [xy[1] for xy in parse_snips_utterance(item)]
    #     print(np.unique(ys))
    if file == 'train':
        domains = np.array([1, 2, 4, 5, 6])
    else:
        domains = np.array([1, 2, 4, 5, 6])

    xy_lists = list()
    np.random.seed(1)

    for i in range(n_tasks):
        xy_lists.append([])
        domain = np.random.choice(domains)
        this_train = total_trains[domain]
        ys = []
        for n, key in enumerate(this_train):
            data = this_train[key]
            for item in data:
                ys += [xy[1] for xy in parse_snips_utterance(item)]

        ys, counts = np.unique(ys, return_counts=True)
        ys[counts < 40] = 'O'
        ys, counts = np.unique(ys, return_counts=True)
        true_tags = []
        n_pairs = 3
        while len(true_tags) < 2*n_pairs:
            random_tag = np.random.choice(ys)
            if random_tag[0] == 'B':
                if len(true_tags)>0 and true_tags[0]==random_tag:
                    continue
                body = random_tag[2:]
                true_tags.append('B-' + body)
                true_tags.append('I-' + body)

        for n, key in enumerate(this_train):
            data = this_train[key]
            for item in data:
                sentence = parse_snips_utterance(item)
                for word in sentence:
                    if word[1] not in true_tags:
                        word[1] = 'O'
                    for index in range(len(true_tags)):
                        if word[1] == true_tags[index]:
                            if index % 2 == 0:
                                word[1] = 'B-' + str(index // 2)
                            else:
                                word[1] = 'I-' + str(index // 2)
                xy_lists[i].append(sentence)
                #print(sentence)
            #1/0
        np.random.shuffle(xy_lists[i])
    #1/0

    # if file == 'train':
    #     super_list = []
    #     for _ in range(task_size):
    #         for task in range(n_tasks):
    #             super_list += list(np.random.choice(xy_lists[task], size=32))
    # else:
    #     # print(len(xy_lists[0]))
    #     # 1/0
    #     if file == 'validate':
    #         super_list = xy_lists[0][:10]
    #     else:
    #         super_list = xy_lists[0][-500:]

    if file == 'train':
        super_list = []
        for _ in range(task_size):
            all_tasks = np.random.choice(n_tasks, size=tasks_batch, replace=False)
            for task in all_tasks:
                super_list += list(np.random.choice(xy_lists[task], size=32 + 32 * 10))
    else:
        # print(len(xy_lists[0]))
        # 1/0
        if file == 'validate':

            super_list = xy_lists[0][:10]
        else:
            super_list = xy_lists[0][-500:]

    return super_list

    1 / 0

    if return_intent:
        return xy_list, intetns
    else:
        return np.array(xy_list)
        # total_train = dict()
        # for folder in contents:
        #     if os.path.isdir(folder):
        #         folder_name = folder.split('/')[-1]
        #         if file=='train':
        #             train_file_name = file + '_' + folder_name + '_full.json'
        #         else:
        #             train_file_name = file + '_' + folder_name + '.json'
        #         with open(os.path.join(folder, train_file_name), encoding='cp1251') as f:
        #             total_train.update(json.load(f))
        #     else:
        #         continue
        #     intetns = list()
        #     xy_list = list()
        #     ys = []
        #     for n, key in enumerate(total_train):
        #         data = total_train[key]
        #         for item in data:
        #             xy_list.append(parse_snips_utterance(item))
        #             #print(xy_list[-1])
        #             # for xx in xy_list[-1]:
        #             #     xx[1]='O'
        #             #1/0
        #             intetns.append(key)
        #             ys += [xy[1] for xy in xy_list[-1]]
        #         # print(xy_list[:10])
        #         # 1/0
        #     #print(ys)
        #     #print(np.unique(ys, return_counts=True))
        # #1/0
        #
        # if return_intent:
        #     return xy_list, intetns
        # else:
        #     return np.array(xy_list)


@DatasetReader.register("snips")
class SnipsDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        if file_path[-8:] == 'test.txt':
            data = snips_reader('test')
        elif file_path[-9:] == 'train.txt':
            data = snips_reader('train')
        else:
            data = snips_reader('validate')
        # if file_path[-9:] == 'train.txt':
        #     print(data[:10])

        for fields in data:
            # unzipping trick returns tuples, but our Fields need lists

            tokens, ner_tags = [list(field) for field in zip(*fields)]
            # TextField requires ``Token`` objects
            tokens = [Token(token) for token in tokens]
            sequence = TextField(tokens, self._token_indexers)

            instance_fields: Dict[str, Field] = {'tokens': sequence}
            # Add "feature labels" to instance
            if 'ner' in self.feature_labels:
                instance_fields['ner_tags'] = SequenceLabelField(ner_tags, sequence, "ner_tags")
            # Add "tag label" to instance
            instance_fields['tags'] = SequenceLabelField(ner_tags, sequence)
            yield Instance(instance_fields)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'SnipsDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return SnipsDatasetReader(token_indexers=token_indexers,
                                  tag_label=tag_label,
                                  feature_labels=feature_labels,
                                  lazy=lazy)
