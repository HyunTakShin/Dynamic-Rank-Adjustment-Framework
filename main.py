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
import numpy as np
import tensorflow as tf
import config as cfg
from train import framework
from solvers.sync_sgd import SyncSGD
from model import (
    resnet20_decomposition,
    LoRAresnet20_decomposition,
    wideresnet28_decomposition,
    LoRAwideresnset28_decomposition
)
from feeders.feeder_cifar import cifar

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    if cfg.dataset == "cifar10":
        batch_size = cfg.cifar10_config["batch_size"]
        num_epochs = cfg.cifar10_config["epochs"]
        min_lr = cfg.cifar10_config["min_lr"]
        max_lr = cfg.cifar10_config["max_lr"]
        num_classes = cfg.cifar10_config["num_classes"]
        decays = list(cfg.cifar10_config["decay"])
        weight_decay = cfg.cifar10_config["weight_decay"]

        dataset = cifar(batch_size = batch_size,
                        num_classes = num_classes)
    elif cfg.dataset == "cifar100":
        batch_size = cfg.cifar100_config["batch_size"]
        num_epochs = cfg.cifar100_config["epochs"]
        min_lr = cfg.cifar100_config["min_lr"]
        max_lr = cfg.cifar100_config["max_lr"]
        num_classes = cfg.cifar100_config["num_classes"]
        decays = list(cfg.cifar100_config["decay"])
        weight_decay = cfg.cifar100_config["weight_decay"]

        dataset = cifar(batch_size = batch_size,
                        num_classes = num_classes)
    else:
        print ("config.py has a wrong dataset definition.\n")
        exit()


    print ("---------------------------------------------------")
    print ("dataset: " + cfg.dataset)
    print ("batch_size: " + str(batch_size))
    print ("training epochs: " + str(num_epochs))
    print ("---------------------------------------------------")
    method = "SVD"
    if cfg.dataset == "cifar10":
        target_layers = []
        ratios = np.full((23), -1).astype(float)
        full_model = resnet20_decomposition(weight_decay, target_layers, ratios).build_model()
        target_layers = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21])
        ratios[target_layers] = 0.5
        partial_model = resnet20_decomposition(weight_decay, target_layers, ratios, method).build_model()
        LoRA_model = LoRAresnet20_decomposition(weight_decay, target_layers, ratios, method).build_model()
        models = [full_model, partial_model, LoRA_model]
    elif cfg.dataset == "cifar100":
        target_layers = []
        ratios = np.full((29), -1).astype(float)
        full_model =wideresnet28_decomposition(weight_decay, target_layers, ratios).build_model()
        target_layers = np.array([0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27])
        ratios[target_layers] = 0.5
        partial_model = wideresnet28_decomposition(weight_decay, target_layers, ratios, method).build_model()
        lora_model = LoRAwideresnset28_decomposition(weight_decay, target_layers, ratios, method).build_model()
        models = [full_model, partial_model, lora_model]
    else:
        print ("Invalid dataset option!\n")
        exit()

    if cfg.optimizer == 0:
        solver = SyncSGD(num_classes = num_classes, models = models, target_layers = target_layers, compression_ratios = ratios)
    else:
        print ("Invalid optimizer option!\n")
        exit()
    trainer = framework(models = models,
                        dataset = dataset,
                        solver = solver,
                        num_epochs = num_epochs,
                        min_lr = min_lr,
                        max_lr = max_lr,
                        decay_epochs = decays,
                        num_classes = num_classes,
                        do_checkpoint = cfg.checkpoint,
                        rank_adjustment = cfg.rank_adjustment)
    trainer.train(method)
