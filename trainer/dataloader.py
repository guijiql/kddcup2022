import os
import pickle
from functools import partial
from datasets import Dataset as HugDataset

import numpy as np
import pandas as pd
import torch


from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import esci2label, task2datapath, cachepath
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import preprocessing

from cocolm.tokenization_cocolm import COCOLMTokenizer
from trainer.sampler import QueryBasedBatchSampler, QueryBasedSampler

tqdm.pandas(desc='apply')

import transformers

transformers.logging.set_verbosity_error()
import re


class Input(object):

    def __init__(self, **kwgs):
        for key, value in kwgs.items():
            value = torch.stack(value, dim=0)
            setattr(self, key, value)

    def cuda(self):
        for key in self.__dict__:
            setattr(self, key, getattr(self, key).cuda())
        return self


class BERTDataLoader(object):

    def __init__(self, config):
        self.config = config
        self._load_data()
        self._split_data()
        self._build_dataset()
        # config.update_value('sample_number', self.sample_number)
        # config.update_value('feature', self.feature)
        # config.update_value('dense_feature', self.dense_feature)
        # config.update_value('sparse_feature', self.sparse_feature)

    def _load_data(self):
        model = self.config.bertmodel.split('/')[-1]
        datapath = os.path.join(cachepath, self.config.task,
                                model + '-' + str(self.config.sample))
        filepath = os.path.join(datapath, 'full_data.pkl')
        if not os.path.exists(filepath):
            self._load_raw_data(datapath)
        else:
            self._load_cache_data(datapath)
        # if self.config.continue_training or self.config.add_task1_data:
        #     self._load_task1_data(datapath)

    def _load_task1_data(self, datapath):
        print("load extra raw data")
        path = task2datapath[self.config.task]
        with open(os.path.join(path, 'product_raw1.pkl'), 'rb') as f:
            product_feat = pickle.load(f)

        # [CLS] [unknown] t-shirt [SEP] [unknown] [SEP]
        columns = ['product_title', 'product_brand', 'product_color_name', 'cate',
                   'product_description', 'product_bullet_point']

        product_feat['product_str'] = product_feat[columns].apply(
            lambda row: ' '.join(row.values.astype(str)).strip(), axis=1
        )
        product_feat.fillna("")

        with open(os.path.join(path, 'task1_extra_query_train_raw.pkl'), 'rb') as f:
            query_train = pickle.load(f)

        query_train['is_train'] = 1
        query_feat = query_train

        # add query ids
        querys = query_feat['query'].values
        query_ids = [0]
        i, ids = 0, self.max_query_ids + 1
        for i in range(1, len(querys)):
            if querys[i] != querys[i - 1]:
                ids += 1
            query_ids.append(ids)

        query_feat['query_id'] = query_ids
        query_feat['query_str'] = query_feat['query']
        query_feat['esci_label'] = query_feat['esci_label'].map(esci2label)

        self.extra_df = pd.merge(query_feat, product_feat,
                                 left_on=['product_id', 'query_locale'],
                                 right_on=['product_id', 'product_locale'],
                                 how='left')
        # save_data
        os.makedirs(datapath, exist_ok=True)
        with open(os.path.join(datapath, 'extra_data.pkl'), 'wb') as f:
            pickle.dump(self.extra_df, f)

    def _load_cache_data(self, datapath):
        print("load cached data")
        with open(os.path.join(datapath, 'full_data.pkl'), 'rb') as f:
            self.df = pickle.load(f)
        self.max_query_ids = np.array(self.df['query_id'].tolist()).max()

    def _load_raw_data(self, datapath):

        print("load raw data")
        path = task2datapath[self.config.task]
        with open(os.path.join(path, 'product_add_cate.pkl'), 'rb') as f:
            product_feat = pickle.load(f)

        # [CLS] [unknown] t-shirt [SEP] [unknown] [SEP]
        columns = ['product_title', 'product_brand', 'product_color_name',
                   'product_description', 'product_bullet_point']

        product_feat['product_str'] = product_feat[columns].apply(
            lambda row: ' '.join(row.values.astype(str)).strip(), axis=1
        )
        product_feat.fillna("")

        with open(os.path.join(path, 'query_train_raw.pkl'), 'rb') as f:
            query_train = pickle.load(f)

        with open(os.path.join(path, 'query_test_raw.pkl'), 'rb') as f:
            query_test = pickle.load(f)
            query_test['esci_label'] = 'exact'  # XXX 为了统一流程

        query_train['is_train'] = 1
        query_test['is_train'] = 0
        query_feat = pd.concat([query_train, query_test], axis=0)

        # add query ids
        querys = query_feat['query'].values
        query_ids = [0]
        i, ids = 0, 0
        for i in range(1, len(querys)):
            if querys[i] != querys[i - 1]:
                ids += 1
            query_ids.append(ids)
        self.max_query_ids = ids

        query_feat['query_id'] = query_ids
        query_feat['query_str'] = query_feat['query']
        query_feat['esci_label'] = query_feat['esci_label'].map(esci2label)

        self.df = pd.merge(query_feat, product_feat,
                           left_on=['product_id', 'query_locale'],
                           right_on=['product_id', 'product_locale'],
                           how='left')
        # save_data
        os.makedirs(datapath, exist_ok=True)
        with open(os.path.join(datapath, 'full_data.pkl'), 'wb') as f:
            pickle.dump(self.df, f)

    def _split_data(self):
        print("split data...")
        trn_val_data = self.df[self.df['is_train'] == 1]
        # extra_trn_data = self.extra_df
        # select locale
        if self.config.locale in ['us', 'jp', 'es']:
            trn_val_data = trn_val_data[trn_val_data['query_locale']
                                        == self.config.locale]
            # extra_trn_data = extra_trn_data[extra_trn_data['query_locale']
            #                                 == self.config.locale]

        # whether sampling
        query_ids = trn_val_data['query_id'].unique()
        if self.config.sample > 0:
            print('sample data...')
            query_ids = np.random.choice(
                query_ids, self.config.sample, replace=False)
        else:
            print('use full data...')
            np.random.shuffle(query_ids)
        n_qs = len(query_ids)
        train_ids = query_ids[:n_qs * 4 // 5]
        valid_ids = query_ids[n_qs * 4 // 5:]
        valid_ids_1 = valid_ids[:len(valid_ids) // 2]  # 用于训练
        valid_ids_2 = valid_ids[len(valid_ids) // 2:]  # 用于valid

        # origin
        # self.train_data = trn_val_data[trn_val_data['query_id'].isin(
        #     train_ids)]
        # self.valid_data = trn_val_data[trn_val_data['query_id'].isin(
        #     valid_ids)]
        # if self.config.continue_training or self.config.add_task1_data:
        #     self.train_data = trn_val_data[trn_val_data['query_id'].isin(
        #         valid_ids_1)]
        #     self.train_data = pd.concat([self.train_data, extra_trn_data], axis=0, ignore_index=True)
        #     self.valid_data = trn_val_data[trn_val_data['query_id'].isin(
        #         valid_ids_2)]
        # if self.config.add_task1_data:
        #     temp = trn_val_data[trn_val_data['query_id'].isin(
        #         train_ids)]
        #     self.train_data = pd.concat([self.train_data, temp], axis=0, ignore_index=True)

        # my version
        self.train_data = trn_val_data[trn_val_data['query_id'].isin(
            train_ids)]
        self.valid_data = trn_val_data[trn_val_data['query_id'].isin(
            valid_ids)]
        import pdb; pdb.set_trace()
        if self.config.continue_training or self.config.add_task1_data:
            self.extra_train_data = trn_val_data[trn_val_data['query_id'].isin(
                valid_ids_1)]
            self.train_data = pd.concat([self.train_data, self.extra_train_data], axis=0, ignore_index=True)
            self.valid_data = trn_val_data[trn_val_data['query_id'].isin(
                valid_ids_2)]

        self.train_data = self.train_data[self.train_data['query_locale'] == 'us']
        self.valid_data = self.valid_data[self.valid_data['query_locale'] == 'us']
        ######

        # trn_val_data = trn_val_data.sample(n=250000)
        # self.train_data, self.valid_data = train_test_split(trn_val_data, test_size=0.2, shuffle=True, random_state=42)
        self.test_data = self.df[self.df['is_train'] == 0]

    def _build_dataset(self):
        self.train_dataset = BERTDataset(
            self.config, dataset=self.train_data, phase='train')
        self.valid_dataset = BERTDataset(
            self.config, dataset=self.valid_data, phase='valid')
        if not self.config.is_offline:  # 非测试环境才会进行测试集的测试
            self.test_dataset = BERTDataset(
                self.config, dataset=self.test_data, phase='test')

    def _build_sampler(self, dataset, max_len=32, shuffle=True):
        assert self.config.sample_strategy.sampler in [
            'query_based', 'record_based']

        batch_sampler = None
        if self.config.sample_strategy.sampler == 'query_based':
            assert self.config.sample_strategy.train_batch_size == 1
            sampler = QueryBasedSampler(dataset.qid2idx_map, shuffle=shuffle)
            batch_sampler = QueryBasedBatchSampler(sampler, 1, max_len=max_len)
        return batch_sampler

    def _collate_fn(self, data, keys):
        feed_dict = {}
        for idx, key in enumerate(keys):
            feed_dict[key] = [tup[idx] for tup in data]
        return Input(**feed_dict)

    def train_dataloader(self):
        batch_sampler = self._build_sampler(self.train_dataset,
                                            max_len=self.config.sample_strategy.max_len,
                                            shuffle=True)
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config.sample_strategy.train_batch_size,
                          batch_sampler=batch_sampler,
                          collate_fn=partial(
                              self._collate_fn, keys=self.train_dataset.keys),
                          shuffle=batch_sampler is None)

    def valid_dataloader(self):
        batch_sampler = self._build_sampler(self.valid_dataset,

                                            max_len=self.config.sample_strategy.max_len,
                                            shuffle=False)
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.config.sample_strategy.eval_batch_size,
                          batch_sampler=batch_sampler,
                          collate_fn=partial(
                              self._collate_fn, keys=self.valid_dataset.keys),
                          shuffle=False)

    def test_dataloader(self):
        batch_sampler = self._build_sampler(self.test_dataset,
                                            max_len=self.config.sample_strategy.max_len,
                                            shuffle=False)
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.config.sample_strategy.eval_batch_size,
                          batch_sampler=batch_sampler,
                          collate_fn=partial(
                              self._collate_fn, keys=self.test_dataset.keys),
                          shuffle=False)


class BERTDataset(Dataset):

    def __init__(self, config, dataset=None, phase="") -> None:
        self.df = dataset
        self.config = config
        self.phase = phase
        self._select_data()
        self._build_map()
        self._build_data()

    def _build_map(self):
        self.qid2idx_map = {}
        for idx, qid in enumerate(self.query_ids):
            if qid not in self.qid2idx_map:
                self.qid2idx_map[qid] = []
            self.qid2idx_map[qid].append(idx)

    def _select_data(self):
        self.query_ids = self.df['query_id'].to_list()
        self.query_str = self.df['query_str'].to_list()
        self.product_str = self.df['product_str'].to_list()
        self.brand = self.df['product_brand'].to_list()
        self.esci_label = self.df['esci_label'].astype(int).to_list()

    def _tokenize(self):
        model = self.config.bertmodel.split('/')[-1]
        datapath = os.path.join(cachepath, self.config.task,
                                model + '-' + str(self.config.sample))
        filepath = os.path.join(datapath, f"{self.phase}_tokenized_data.pkl")

        def _load_tokenized_data(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        def encode(batch):
            return self.tokenizer(batch["query_str"], batch["product_str"],
                                  add_special_tokens=True,
                                  max_length=512,
                                  truncation=True,
                                  return_attention_mask=True,
                                  padding='max_length')

        if os.path.exists(filepath):
            return _load_tokenized_data(filepath)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.bertmodel, do_lower_case=True, use_fast=True)

            self.str_df = HugDataset.from_dict({"query_str": self.query_str, "product_str": self.product_str})
            tokenize_dataset = self.str_df.map(
                encode,
                batched=True,
                remove_columns=['query_str', 'product_str'],
                load_from_cache_file=False,
                num_proc=1,  # 多进程加速
                desc="Running tokenizer on dataset",
            ).to_dict()

            with open(filepath, 'wb') as f:
                pickle.dump(tokenize_dataset, f)
            return tokenize_dataset
# =======
#     def encode(self, batch):
#         return self.tokenizer(batch["query_str"], batch["product_str"],
#                               add_special_tokens=True,
#                               max_length=512,
#                               truncation=True,
#                               return_attention_mask=True,
#                               padding='max_length')
#
#     def _dt_tokenize(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.bertmodel, do_lower_case=True, use_fast=True)
#
#         self.query_df = HugDataset.from_dict({"text": self.query_str})
#         self.product_df = HugDataset.from_dict({"text": self.product_str})
#         query_tokenize_dataset = self.query_df.map(
#             self.encode,
#             batched=True,
#             remove_columns=['text'],
#             load_from_cache_file=False,
#             num_proc=4,  # 多进程加速
#             desc="Running tokenizer on query dataset",
#         )
#
#         product_tokenize_dataset = self.product_df.map(
#             self.encode,
#             batched=True,
#             remove_columns=['text'],
#             load_from_cache_file=False,
#             num_proc=4,  # 多进程加速
#             desc="Running tokenizer on product dataset",
#         )
#
#         return query_tokenize_dataset.to_dict(), product_tokenize_dataset.to_dict()
#
#     def _tokenize(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.bertmodel, do_lower_case=True, use_fast=True)
#         self.str_df = HugDataset.from_dict({"query_str": self.query_str, "product_str": self.product_str})
#         tokenize_dataset = self.str_df.map(
#             self.encode,
#             batched=True,
#             remove_columns=['query_str', 'product_str'],
#             load_from_cache_file=False,
#             num_proc=4,  # 多进程加速
#             desc="Running tokenizer on dataset",
#         )
#
#         return tokenize_dataset.to_dict()  # , tokenize_brand['input_ids']
# >>>>>>> Stashed changes

    def _add_attr(self, key, val):
        self.keys.append(key)
        self.tensors.append(val)

    def _match(self, data):
        brand_flag = torch.zeros(len(data['input']))
        for i in range(0, len(data['input'])):
            if data['input'][i: i + len(data['brand'])] == data['brand']:
                brand_flag[i: i + len(data['brand'])] = 1
        return brand_flag

    def _build_data(self):
        self.keys = []
        self.tensors = []
        tokenize_data = self._tokenize()
        self._add_attr('query_id', torch.tensor(self.query_ids))
        for key, val in tokenize_data.items():
            self._add_attr(key, torch.tensor(val))
        self._add_attr('esci_label', torch.tensor(self.esci_label))
        # self._add_attr('inputs_embeds', torch.tensor(self.esci_label))
        # for key, val in query_tokenize_data.items():
        #     self._add_attr(f"query_{key}", torch.tensor(val))
        # for key, val in product_tokenize_data.items():
        #     self._add_attr(f"product_{key}", torch.tensor(val))
        # self._add_attr('esci_label', torch.tensor(self.esci_label))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
