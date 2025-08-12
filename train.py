'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/03/17
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/08/11
Hyuntak Shin, B.A Participant.
<hyuntakshin@inha.ac.kr>
'''

import time
import math
import random
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
from copy import deepcopy
from tensorflow.keras.metrics import Mean


class framework:
    def __init__ (self, models, dataset, solver, **kargs):
        self.dataset = dataset
        self.solver = solver
        self.num_epochs = kargs["num_epochs"]
        self.min_lr = kargs["min_lr"]
        self.max_lr = kargs["max_lr"]
        self.decay_epochs = kargs["decay_epochs"]
        self.do_checkpoint = kargs["do_checkpoint"]
        self.num_classes = kargs["num_classes"]
        self.rank_adjustment = kargs["rank_adjustment"]
        self.warmup_epochs = 0
        self.target = 0
        self.lr_decay_factor = 10
        if self.num_classes == 1:
            self.valid_acc = tf.keras.metrics.BinaryAccuracy()
        else:
            self.valid_acc = tf.keras.metrics.Accuracy()

        self.checkpoint = tf.train.Checkpoint(models = models, optimizers = self.solver.optimizers)
        for i in range (len(self.checkpoint.optimizers)):
            self.checkpoint.optimizers[i].lr.assign(self.min_lr)
        checkpoint_dir = "./checkpoint"
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint, directory = checkpoint_dir, max_to_keep = 3)
        self.checkpoint.models[0].summary()
        self.checkpoint.models[1].summary()
        self.checkpoint.models[2].summary()

        # Resume if any checkpoints are in the current directory.
        self.resume()

    def resume (self):
        self.epoch_id = 0
        if self.checkpoint_manager.latest_checkpoint:
            self.epoch_id = int(self.checkpoint_manager.latest_checkpoint.split(sep='ckpt-')[-1]) - 1
            print ("Resuming the training from epoch %3d\n" % (self.epoch_id))
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

            if self.epoch_id < self.rank_adjustment[0]:
                self.target = 1
            elif self.epoch_id < self.rank_adjustment[1]:
                self.target = 0
            else:
                self.target = 2

    def train (self, method):
        train_dataset = self.dataset.train_dataset()
        valid_dataset = self.dataset.valid_dataset()

        # Calculate the warmup lr increase.
        warmup_step_lr = 0
        if self.warmup_epochs > 0:
            num_warmup_steps = self.warmup_epochs * self.average_interval
            warmup_step_lr = (self.max_lr - self.min_lr) / num_warmup_steps

        # Broadcast the parameters from rank 0 at the first epoch.
        start_epoch = self.epoch_id

        for epoch_id in range (start_epoch, self.num_epochs):
            lossmean = Mean()

            # LR decay
            if epoch_id in self.decay_epochs:
                lr_decay = 1 / self.lr_decay_factor
                for i in range (len(self.checkpoint.optimizers)):
                    self.checkpoint.optimizers[i].lr.assign(self.checkpoint.optimizers[i].lr * lr_decay)
            if epoch_id == 0:
                self.target = 1
            if epoch_id == self.rank_adjustment[0]:
                print ("Inflating the model...\n")
                self.target = 0
                if method == "SVD":
                    self.solver.recover_model(self.checkpoint, 1, self.target)
                elif method == "CP":
                    self.solver.recover_cp_model(self.checkpoint, 1, self.target)
                elif method == "Tucker":
                    self.solver.recover_tucker_model(self.checkpoint, 1, self.target)
            elif epoch_id == self.rank_adjustment[1]:
                print ("Deflating the model...\n")
                self.target = 2
                if method == "SVD":
                    self.solver.toLoRA(self.checkpoint, 0, self.target)
                elif method == "CP":
                    self.solver.toCPLoRA(self.checkpoint, 0, self.target)
                elif method == "Tucker":
                    self.solver.toTuckerLoRA(self.checkpoint, 0, self.target)
            
            # Training loop.
            lossmean.reset_states()
            for j in tqdm(range(self.dataset.num_train_batches), ascii=True):
                if epoch_id < self.warmup_epochs:
                    for i in range (len(self.checkpoint.optimizers)):
                        self.checkpoint.optimizers[i].lr.assign(self.checkpoint.optimizers[i].lr + warmup_step_lr)
                images, labels = train_dataset.next()
                loss, grads = self.solver.train_step(self.checkpoint, images, labels, self.target)
                lossmean(loss)

            # Collect the global training results (loss and accuracy).
            local_loss = lossmean.result().numpy()

            # Collect the global validation accuracy.
            local_acc = self.evaluate(valid_dataset, self.target)
            dir = "./"
            print(f"Epoch {int(epoch_id)} "
                        f"target model: {self.target} "
                        f"lr: {self.checkpoint.optimizers[self.target].lr.numpy()} "
                        f"validation acc = {local_acc} "
                        f"training loss = {local_loss}")
            f = open(dir+"acc.txt", "a")
            f.write(str(local_acc) + "\n")
            f.close()
            f = open(dir+"loss.txt", "a")
            f.write(str(local_loss) + "\n")
            f.close()

            # Checkpointing
            if self.do_checkpoint == True:
                self.checkpoint_manager.save()

    def evaluate (self, valid_dataset, target):
        self.valid_acc.reset_states()
        for i in tqdm(range(self.dataset.num_valid_batches), ascii=True):
            data, label = valid_dataset.next()
            predicts = self.checkpoint.models[target](data)
            if len(label.shape) == 1:
                self.valid_acc(label, predicts)
            else:
                self.valid_acc(tf.argmax(label, 1), tf.argmax(predicts, 1))
        accuracy = self.valid_acc.result().numpy()
        return accuracy
