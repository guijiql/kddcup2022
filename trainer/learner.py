import datetime
from genericpath import exists
import glob
import re
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from utils import Logger, dict2str, label2esci, savepath, optim_dict

from models import PGD, FGM, FreeLB, AWP
from torch.cuda.amp import autocast, GradScaler


class BaseLearner(object):

    def __init__(self, config, model_cls, loader):
        self.loader = loader
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
        self.model = model_cls(config)

        self.model.cuda()

        self.epochs = config.epochs
        self.early_stop_rounds = config.early_stop
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

        self.cur_epoch = 0
        self.cur_step = 0
        self.best_epoch = 0
        self.best_result = -np.inf

        param_optimizer_bert = list(self.model.named_parameters())
        no_decay_bert = ['bias', 'gamma', 'beta']

        optimizer_grouped_parameters_bert = \
            [
                {'params': [p for n, p in param_optimizer_bert if not any(nd in n for nd in no_decay_bert) and p.requires_grad],
                 'weight_decay': self.weight_decay, 'lr': self.learning_rate},
                {'params': [p for n, p in param_optimizer_bert if any(nd in n for nd in no_decay_bert) and p.requires_grad],
                 'weight_decay': 0.0, 'lr': self.learning_rate}
            ]
        self.optimizer = getattr(torch.optim, config.optimizer)(optimizer_grouped_parameters_bert,
                                                                lr=self.learning_rate, eps=1e-2)

        self.save_floder = os.path.join(
            savepath, config.task, config.model, config.commit)
        self.submit_floder = os.path.join(
            savepath, config.task, config.model, config.commit, "submit")
        os.makedirs(self.save_floder, exist_ok=True)
        os.makedirs(self.submit_floder, exist_ok=True)

        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.writer = SummaryWriter(self.save_floder, comment=date)
        self.logger = Logger(os.path.join(self.save_floder, 'run.log'))

        self.pgd = PGD(self.model)
        self.fgm = FGM(self.model)
        self.pgd_k = 2
        self.freelb = FreeLB()
        self.scaler = GradScaler()
        self.awp = AWP(self.model, self.optimizer, apex=config.apex, adv_lr=config.awp_lr, adv_eps=config.awp_eps)
        self.load_model()
        # self.fast_awp = FastAWP(self.model, self.optimizer, adv_lr=0.001, adv_eps=0.001)

    def train_one_epoch(self):
        print(f"epoch {self.cur_epoch}, step {self.eval_step}")
        loader = self.loader.train_dataloader()
        total_loss = []
        with tqdm.tqdm(loader, ncols=100) as pbar:
            for input in pbar:
                if self.cur_step > self.last_step:
                    if (self.cur_step + 1) % self.eval_step == 0:
                        self.valid()
                    train_loss = self.train_one_step(input)
                    total_loss.append(train_loss)
                    train_loss = self.smooth_loss(total_loss)
                    pbar.set_description(
                        f'train loss: {train_loss:.4f}')
                    self.writer.add_scalar(
                        'train/train_loss', train_loss, global_step=self.cur_step)
                self.cur_step += 1

    def smooth_loss(self, losses):
        if len(losses) < 100:
            return sum(losses) / len(losses)
        else:
            return sum(losses[-100:]) / 100

    def train_one_step_pgd(self, input):
        self.model.train()
        input = input.cuda()
        loss = self.model.calculate_loss(input)
        loss.backward()
        self.pgd.backup_grad()
        for i in range(self.pgd_k):
            self.pgd.attack(is_first_attack=(i == 0))
            if i != self.pgd_k - 1:
                self.optimizer.zero_grad()
            else:
                self.pgd.restore_grad()
            loss_adv = self.model.calculate_loss(input)
            loss_adv.backward()
        self.pgd.restore()
        train_loss = loss_adv.item()
        self.optimizer.step()
        self.model.zero_grad()
        return train_loss

    def train_one_step_fgm(self, input):
        self.model.train()
        input = input.cuda()
        loss = self.model.calculate_loss(input)
        loss.backward()
        self.fgm.attack()
        loss_adv = self.model.calculate_loss(input)
        loss_adv.backward()
        self.fgm.restore()
        train_loss = loss_adv.item()
        self.optimizer.step()
        self.model.zero_grad()
        return train_loss

    def train_one_step_freelb(self, input):
        self.model.train()
        input = input.cuda()
        loss = self.freelb.attack(self.model, input)
        self.optimizer.step()
        self.model.zero_grad()
        return loss.item()

    def train_one_step_awp(self, input):
        self.model.train()
        input = input.cuda()
        loss = self.model.calculate_loss(input)
        self.optimizer.zero_grad()
        loss.backward()

        if self.cur_epoch >= self.config.start_epoch:
            loss = self.awp.attack_backward(input)
            loss.backward()
            self.awp._restore()

        self.optimizer.step()

        return loss.item()

    def train_one_step_fast_awp(self, input):
        self.model.train()
        input = input.cuda()
        loss = self.model.calculate_loss(input)
        self.optimizer.zero_grad()
        loss.backward()

        if self.cur_epoch >= self.config.start_epoch:
            self.fast_awp.perturb(input)
            self.fast_awp.restore()

        self.optimizer.step()

        return loss.item()

    def train_one_step_apex(self, input):
        self.model.train()
        input = input.cuda()
        with torch.cuda.amp.autocast(enabled=self.config.apex):
            loss = self.model.calculate_loss(input)

        self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        self.scaler.scale(loss).backward()
        for name, para in self.model.named_parameters():
            try:
                if not torch.isnan(para.grad).all():
                    import pdb; pdb.set_trace()
            except:
                print("NoneType")
                import pdb; pdb.set_trace()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # import pdb; pdb.set_trace()

        return loss.item()

    def train_one_step(self, input):
        self.model.train()
        input = input.cuda()
        loss = self.model.calculate_loss(input)
        train_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_loss

    def train(self):
        train_length = len(self.loader.train_dataloader())
        eval = [4, 4, 4, 5, 5]
        while self.cur_epoch < self.epochs:
            self.eval_step = train_length // eval[self.cur_epoch]
            self.train_one_epoch()
            if self.cur_epoch - self.best_epoch > self.early_stop_rounds:
                self.logger.info("early stop...")
                break
            self.cur_epoch += 1
        self.writer.add_hparams(self.config.to_parm_dict(), {
            'valid/best_result': self.best_result})

    def evaluate(self, pred_dict):
        ans = {}
        ans['ACC'] = accuracy_score(pred_dict['y_trues'], pred_dict['y_preds'])

        df = pd.DataFrame({
            'y_preds': pred_dict['y_preds'].tolist(),
            'y_trues': pred_dict['y_trues'].tolist(),
            'y_probs': pred_dict['y_probs'].tolist(),
            'query_locale': self.loader.valid_data['query_locale'].to_list()
        })

        for locale, sub_df in df.groupby('query_locale'):
            ans[f'{locale}_ACC'.upper()] = accuracy_score(
                sub_df['y_trues'], sub_df['y_preds'])
        return ans

    def valid(self):
        pred_dict = self.eval_one_epoch(
            loader=self.loader.valid_dataloader())
        metrics = self.evaluate(pred_dict)
        self.logger.info(
            f'[epoch={self.cur_epoch}, step={self.cur_step}] {dict2str(metrics)}')
        if metrics['ACC'] > self.best_result:
            self.best_result = metrics['ACC']
            self.best_epoch = self.cur_epoch
            self.best_step = self.cur_step
            self.save_model()
            self.ans2csv(self.loader.valid_data, pred_dict,
                         phase='valid')
        for key, value in metrics.items():
            self.writer.add_scalar(
                f'valid/{key}', value, global_step=self.cur_step)
        return metrics

    def test(self):
        self.load_model()
        pred_dict = self.eval_one_epoch(
            loader=self.loader.test_dataloader())
        self.ans2csv(self.loader.test_data, pred_dict, phase='test')

    def ans2csv(self, data, pred_dict, phase=None):
        assert phase in ['valid', 'test']
        ans = data.copy()
        ans['esci_label_name'] = ans['esci_label'].map(label2esci)

        ans['esci_label_pred'] = pred_dict['y_preds'].tolist()
        ans["esci_probs"] = pred_dict['y_probs'].tolist()
        ans["esci_label_pred_name"] = ans["esci_label_pred"].map(label2esci)

        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        ans.to_csv(os.path.join(self.submit_floder,
                                f'{date}_{phase}_{self.cur_step}_{self.best_result:.5f}.csv'), index=False)

    @torch.no_grad()
    def eval_one_epoch(self, loader):
        self.model.eval()
        pred_dict = {
            "y_preds":[],
            "y_probs": [],
            "y_trues": []
        }
        with tqdm.tqdm(loader, ncols=100) as pbar:
            for input in pbar:
                input = input.cuda()
                with torch.cuda.amp.autocast(enabled=True):
                    scores = self.model.predict(input)
                pred_dict["y_trues"].append(input.esci_label)
                pred_dict["y_preds"].append(scores.argmax(dim=1))
                pred_dict["y_probs"].append(scores)
            for key, val in pred_dict.items():
                pred_dict[key] = torch.cat(val, dim=0).cpu().numpy()
        return pred_dict

    def save_model(self):
        filename = os.path.join(
            self.save_floder,
            f"epoch={self.cur_epoch}-step={self.cur_step}-acc={self.best_result}.pth")
        state = {
            'best_epoch': self.best_epoch,
            'cur_epoch': self.cur_epoch,
            'cur_step': self.cur_step,
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
        }
        torch.save(state, filename)

    def load_model(self):
        # use the best pth
        modelfiles = glob.glob(os.path.join(self.save_floder, '*.pth'))
        accuracys = [re.search("acc=(.*).pth", file).group(1)
                     for file in modelfiles]
        accuracys = [float(acc) for acc in accuracys]
        idx = np.argmax(accuracys)

        state = torch.load(modelfiles[idx])
        print(f"loading {modelfiles[idx]} ...")
        # self.cur_epoch = state['cur_epoch']
        # self.cur_step = state['cur_step']
        # self.best_epoch = state['best_epoch']
        self.last_step = state['cur_step']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer_dict'])
