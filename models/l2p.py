import logging
import numpy as np
import torch
import torch.distributed as dist

from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.inc_net import L2PNet
from utils.prompt import Prompt
from models.base import BaseLearner
from pathlib import Path
import timm
from timm.models.registry import register_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.models import create_model
from vision_transformer import _create_vision_transformer
from utils.toolkit import target2onehot, tensor2numpy
import argparse
import os
import random
import torch.backends.cudnn as cudnn

init_epoch = 5
epochs =5
num_workers = 8

class L2P(BaseLearner):
    def __init__(self, json_file):
        super().__init__(json_file)
        parser = argparse.ArgumentParser('L2P training and evaluation configs')
        
        if json_file["dataset"] == 'cifar100':
            from configs.cifar100_l2p import get_args_parser
        else:
            raise NotImplementedError
        
        get_args_parser(parser)
        args = parser.parse_args()
        logging.info('L2P original args:{}'.format(args))
        self.args = args
        args_dict = vars(self.args)
        args_dict.update(json_file)
        self.args.distributed = False
        
        self._network = L2PNet(self.args,True)
        if self.args.unscale_lr:
            global_batch_size = self.args.batch_size
        else:
            global_batch_size = self.args.batch_size * self.args.world_size
        self.args.lr = self.args.lr * global_batch_size / 256.0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        data_manager._train_trsf = self.trans_data(True)
        data_manager._test_trsf = self.trans_data(False)
        data_manager._common_trsf = []
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
        )

        n_parameters = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)    

        if self._cur_task == 0:
            optimizer = create_optimizer(self.args, self._network)
            if self.args.sched != 'constant':
                self.scheduler, _ = create_scheduler(self.args, optimizer)
            elif self.args.sched == 'constant':
                self.scheduler = None
            self._init_train(train_loader, test_loader, optimizer, self.scheduler)
        else:
            if self.args.reinit_optimizer:
                optimizer = create_optimizer(self.args, self._network)
            self._update_representation(train_loader, test_loader, optimizer, self.scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs, task_id=self._cur_task)
                logits = output['logits']
                logits = logits.index_fill(dim=1, index=self.fc_mask().to(self._device), value=float('-inf'))

                losses = F.cross_entropy(logits, targets.long()).to(self._device)
                if self.args.pull_constraint and 'reduce_sim' in output:
                    losses = losses - self.args.pull_constraint_coeff * output['reduce_sim']
                
                optimizer.zero_grad()
                losses.backward()

                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=self.args.clip_grad)
                optimizer.step()
                torch.cuda.synchronize()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step(epoch)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if self.args.prompt_pool and self.args.shared_prompt_pool:
              
            prev_start = (self._cur_task - 1) * self.args.top_k
            prev_end = self._cur_task  * self.args.top_k

            cur_start = prev_end
            cur_end = (self._cur_task  + 1) * self.args.top_k

            if (prev_end > self.args.size) or (cur_end > self.args.size):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

                with torch.no_grad():
                    if self.args.distributed:
                        self._network.module.prompt.prompt.grad.zero_()
                        self._network.module.prompt.prompt[cur_idx] = self._network.module.prompt.prompt[prev_idx]
                        optimizer.param_groups[0]['params'] = self._network.module.parameters()
                    else:
                        self._network.prompt.prompt.grad.zero_()
                        self._network.prompt.prompt[cur_idx] = self._network.prompt.prompt[prev_idx]
                        optimizer.param_groups[0]['params'] = self._network.parameters()
                        
            # Transfer previous learned prompt param keys to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_key:
            
            prev_start = (self._cur_task  - 1) * self.args.top_k
            prev_end = self._cur_task  * self.args.top_k

            cur_start = prev_end
            cur_end = (self._cur_task  + 1) * self.args.top_k

            with torch.no_grad():
                if self.args.distributed:
                    self._network.module.prompt.prompt_key.grad.zero_()
                    self._network.module.prompt.prompt_key[cur_idx] = self._network.module.prompt.prompt_key[prev_idx]
                    optimizer.param_groups[0]['params'] = self._network.module.parameters()
                else:
                    self._network.prompt.prompt_key.grad.zero_()
                    self._network.prompt.prompt_key[cur_idx] = self._network.prompt.prompt_key[prev_idx]
                    optimizer.param_groups[0]['params'] = self._network.parameters()

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                self._network.train()
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self._network(inputs, self._cur_task)
                logits = output["logits"]
                logits = logits.index_fill(dim=1, index=self.fc_mask().to(self._device), value=float('-inf'))

                loss = F.cross_entropy(logits, targets) # base criterion (CrossEntropyLoss)
                if self.args.pull_constraint and 'reduce_sim' in output:
                    loss = loss - self.args.pull_constraint_coeff * output['reduce_sim']

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.args.clip_grad)
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            if scheduler:
                scheduler.step(epoch)

            #scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def fc_mask(self):
        class_mask = np.arange(self._known_classes,self._total_classes)
        not_mask = np.setdiff1d(np.arange(self.args.nb_classes),class_mask)
        #self.not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self._device)
        not_mask = torch.tensor(not_mask, dtype=torch.int64)
        return not_mask

    def trans_data(self,is_train):
        resize_im = self.args.input_size > 32
        if is_train:
            scale = (0.05, 1.0)
            ratio = (3. / 4., 4. / 3.)
            
            return  [
                transforms.RandomResizedCrop(self.args.input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * self.args.input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(self.args.input_size))
        t.append(transforms.ToTensor())
        
        return t
        #return transforms.Compose(t)

@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model