import argparse
import os
import random

import numpy as np
import torch

import models
from utils import Config, seed_everything, make_dirs
from trainer import BaseLearner, BERTDataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--commit', type=str, required=True)
parser.add_argument('--locale', type=str, default='all',
                    choices=['all', 'us', 'jp', 'es'])
parser.add_argument('--sample', type=int, default=-1)
parser.add_argument('--model', type=str, default="DeBERTa")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--task', type=str, default="task2")
parser.add_argument('--is_offline', type=int, default=1)

args = parser.parse_args()


if __name__ == "__main__":

    seed_everything(2048)
    args = Config(args)
    make_dirs()
    config = args.easy_use()
    print(config.__dict__)

    loader = BERTDataLoader(config)
    
    model_cls = getattr(models, config.model)
    learner = BaseLearner(config, model_cls, loader)
    args.tab_printer(learner.logger)
    learner.train()
    if not config.is_offline:
        learner.test()
