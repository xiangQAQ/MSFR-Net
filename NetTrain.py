import torch
import numpy as np
from _warnings import warn
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
matplotlib.use("agg")
from time import time, sleep
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime

class NetworkTrainer(object):
    def __init__(self, deterministic=True):

        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        if deterministic:
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        self.loss = None

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 10
        self.val_eval_criterion_alpha = 0.95  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.95  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.save_every = 5
        self.save_latest_only = True
        self.max_num_epochs = 500
        self.num_batches_per_epoch = 58
        self.num_val_batches_per_epoch = 7
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
    @abstractmethod
    def initialize(self, training=True):
        print('xx')

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            # mkdir_if_not_exist(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                         (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)


    def run_iteration(self, data_generator, do_backprop=True):
        data_dict = next(data_generator)
        data = data_dict['data']
        seg = data_dict['seg']
        seg_t1c = seg.copy()
        seg_flair = seg.copy()
        seg_t1c[np.where(seg_t1c == 2)] = 0
        seg_flair[np.where(seg_flair == 1)] = 3

        data = torch.from_numpy(data).float()
        seg_all = torch.from_numpy(seg).float()
        seg_t1c = torch.from_numpy(seg_t1c).float()
        seg_flair = torch.from_numpy(seg_flair).float()

        data = data.cuda(non_blocking=True)
        seg_all  = seg_all.cuda(non_blocking=True)
        seg_t1c = seg_t1c.cuda(non_blocking=True)
        seg_flair = seg_flair.cuda(non_blocking=True)

        self.optimizer.zero_grad()

        all_out, t1c_out, flair_out = self.network(data)
        del data
        all_l = self.loss(all_out, seg_all)
        t1c_l = self.loss(t1c_out, seg_t1c)
        flair_l = self.loss(flair_out, seg_flair)
        l = all_l*0.7+t1c_l*0.15+flair_l*0.15
        del seg_all, seg_t1c, seg_flair

        if do_backprop:
            l.backward()
            self.optimizer.step()

        return l.detach().cpu().numpy(), all_l.detach().cpu().numpy(), t1c_l.detach().cpu().numpy(), flair_l.detach().cpu().numpy()

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                     self.all_tr_losses[-1]

    def plot_progress(self):
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(self.all_val_losses):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def maybe_update_lr(self):
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics)},
            fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def maybe_save_checkpoint(self):
        if self.epoch % self.save_every == (self.save_every - 1):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower 
                is better, so we need to negate it. 
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience,
                                        self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                self.print_to_log_file(
                    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def on_epoch_end(self):

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()

        return continue_training

    def run_training(self):
        torch.cuda.empty_cache()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        #mkdir_if_not_exist(self.output_folder)

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            tr_l_epoch = []
            tr_all_l_epoch = []
            tr_t1c_l_epoch = []
            tr_flair_l_epoch = []
            # train one epoch
            self.network.train()
            for b in range(self.num_batches_per_epoch):
                tr_l, tr_all_l, tr_t1c_l, tr_flair_l = self.run_iteration(self.dl_tr, True)
                tr_l_epoch.append(tr_l)
                tr_all_l_epoch.append(tr_all_l)
                tr_t1c_l_epoch.append(tr_t1c_l)
                tr_flair_l_epoch.append(tr_flair_l)
            self.all_tr_losses.append(np.mean(tr_l_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1],
                                    "train all_loss : %.4f" % np.mean(tr_all_l_epoch),
                                   "train t1c_loss : %.4f" % np.mean(tr_t1c_l_epoch),
                                   "train flair_loss : %.4f" % np.mean(tr_flair_l_epoch))

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_l_epoch = []
                val_all_l_epoch = []
                val_t1c_l_epoch = []
                val_flair_l_epoch = []
                for b in range(self.num_val_batches_per_epoch):
                    val_l, val_all_l, val_t1c_l, val_flair_l = self.run_iteration(self.dl_val, False)
                    val_l_epoch.append(val_l)
                    val_all_l_epoch.append(val_all_l)
                    val_t1c_l_epoch.append(val_t1c_l)
                    val_flair_l_epoch.append(val_flair_l)
                self.all_val_losses.append(np.mean(val_l_epoch))
                self.print_to_log_file("val loss (train=False): %.4f" % self.all_val_losses[-1],
                                       "val all_loss : %.4f" % np.mean(val_all_l_epoch),
                                       "val t1c_loss : %.4f" % np.mean(val_t1c_l_epoch),
                                       "val flair_loss : %.4f" % np.mean(val_flair_l_epoch)                                       )

            epoch_end_time = time()
            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training
            continue_training = self.on_epoch_end()
            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))