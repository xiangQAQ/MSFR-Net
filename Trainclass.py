from dataloader import DataLoader3D
from utilities import softmax_helper, InitWeights_He
import torch
from torch import nn
import numpy as np
from torch.optim import lr_scheduler
from loss import DC_and_CE_loss
#from GRUNet import GRU_UNet
from MSnet import MM_net
from NetTrain import NetworkTrainer
from new_dataload import get_dataload
from collections import OrderedDict


class MTrainer(NetworkTrainer):
    def __init__(self, output_folder=None, train_dir=None, val_dir=None, deterministic=True):
        super(MTrainer, self).__init__(deterministic)
        # set through arguments from init
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_folder = output_folder
        self.gpu_ids = [0, 1, 2, 3]
        self.dl_tr = self.dl_val = None
        self.patch_size = (96, 128, 128)
        self.batch_size = 4
        self.batch_dice = False
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {})

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 10
        self.initial_lr = 1e-4
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33


    def initialize(self, training=True):
        #mkdir_if_not_exist(self.output_folder)
        if training:
            self.dl_tr, self.dl_val = get_dataload(self.train_dir, self.val_dir, self.patch_size, self.batch_size)
        self.network = MM_net()
        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        saved_model = torch.load('output1/model_best.model',
                                 map_location='cuda:0')

        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr,
                                              weight_decay=self.weight_decay,
                                              amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                               patience=self.lr_scheduler_patience,
                                                               verbose=True, threshold=self.lr_scheduler_eps,
                                                               threshold_mode="abs")
        self.network.to(self.gpu_ids[0])
        self.network = torch.nn.DataParallel(self.network, self.gpu_ids)  # multi-GPUs
        #self.network.cuda()

        self.was_initialized = True

    def run_training(self):
        super(MTrainer, self).run_training()