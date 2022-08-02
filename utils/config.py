import json
import os
import torch
from texttable import Texttable
from .logger import Logger
from .setting import confpath

class EasyDict():
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, EasyDict(v))
            else:
                setattr(self, k, v)

    def to_parm_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, torch.Tensor)):
                result[key] = value
            elif isinstance(value, EasyDict):
                for lkey, lvalue in value.__dict__.items():
                    if isinstance(lvalue, (str, int, float, bool, torch.Tensor)):
                        result[key + '.' + lkey] = lvalue
        return result


class Config(object):

    def __init__(self, args):
        self.confpath = os.path.join(confpath, args.task, f'{args.model}.json')

        self.config_in_files = self.load_config()
        self.config_in_cmd = vars(args)

        self.config = self.merge_config()

    def easy_use(self):
        return EasyDict(self.config)

    def load_config(self):
        return json.load(open(self.confpath, 'r'))

    def merge_config(self):
        config = {}
        config.update(self.config_in_files)
        config.update(self.config_in_cmd)
        return config

    def tab_printer(self, logger):
        """
        Function to print the logs in a nice tabular format.
        :param args: Parameters used for the model.
        """
        args = self.easy_use().to_parm_dict()
        keys = sorted(args.keys())
        t = Texttable(max_width=100)
        t.set_precision(10)
        t.add_rows([["Parameter", "Value"]] +
                   [[k.capitalize(), str(args[k])] for k in keys])
        logger.info('\n' + t.draw())
