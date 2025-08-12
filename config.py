'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/08/11
Hyuntak Shin, B.A Participant.
<hyuntakshin@inha.ac.kr>
Notice
 - "batch_size" is a local batch size per process.
 - The total batch size is the # of processes multiplied by the batch size.
'''

cifar10_config = {
    "batch_size": 128,
    "min_lr": 0.1,
    "max_lr": 0.1,
    "num_classes": 10,
    "epochs": 150,
    "decay": {100, 130},
    "rank_adjustment":{55,120},
    "weight_decay": 0.0001,
}

cifar100_config = {
    "batch_size": 128,
    "min_lr": 0.1,
    "max_lr": 0.1,
    "num_classes": 100,
    "epochs": 200,
    "decay": {150,180},
    "rank_adjustment": {80, 170},
    "weight_decay": 0.0005,
}

num_processes_per_node = 8
dataset = "cifar100"
checkpoint = 1
rank_adjustment = [55,120]
optimizer = 0
