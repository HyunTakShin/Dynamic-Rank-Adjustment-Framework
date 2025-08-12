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
import os
import random
import time
import math
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE
import tensorflow_datasets as tfds

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=float)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

class cifar:
    def __init__ (self, batch_size, num_classes):
        self.num_classes = num_classes
        self.train_batch_size = batch_size
        self.valid_batch_size = 125
        train_data = None
        train_label = None

        # Load the data splits.
        name = "cifar" + str(self.num_classes)
        dataset = tfds.as_numpy(tfds.load(name = name, split = tfds.Split.TRAIN, batch_size = -1))
        train_data, train_label = dataset["image"], dataset["label"]
        dataset = tfds.as_numpy(tfds.load(name = name, split = tfds.Split.TEST, batch_size = -1))
        valid_data, valid_label = dataset["image"], dataset["label"]

        self.train_data = train_data.astype('float32') / 255.
        self.valid_data = valid_data.astype('float32') / 255.

        self.train_label = dense_to_one_hot(train_label, num_classes)
        self.valid_label = dense_to_one_hot(valid_label, num_classes)

        self.num_train_batches = int(math.floor(50000 / (self.train_batch_size)))
        self.num_valid_batches = int(math.floor(10000 / (self.valid_batch_size)))

        self.num_local_train_samples = self.num_train_batches * self.train_batch_size
        self.num_local_valid_samples = self.num_valid_batches * self.valid_batch_size
        self.num_valid_samples = self.num_valid_batches * self.valid_batch_size
        self.train_sample_offset = 0 * self.num_local_train_samples
        self.valid_sample_offset = 0 * self.num_local_valid_samples

        print ("Number of training batches: " + str(self.num_train_batches))
        print ("Number of validation batches: " + str(self.num_valid_batches))

        self.per_pixel_mean = np.array(self.train_data).astype(np.float32).mean(axis=0)
        self.per_pixel_std = np.array(self.train_data).astype(np.float32).std(axis=0)
        self.shuffle()
        
    def shuffle (self):
        self.shuffled_index = np.arange(50000, dtype='int32')
        random.Random(time.time()).shuffle(self.shuffled_index)

    def read_train_image (self, sample_id):
        index = self.shuffled_index[self.train_sample_offset + sample_id.numpy()]
        image = self.train_data[index]
        image = np.subtract(image, self.per_pixel_mean)
        image = np.divide(image, self.per_pixel_std)
        label = self.train_label[index]
        return image, label

    def read_valid_image (self, sample_id):
        index = sample_id.numpy()
        image = self.valid_data[index]
        image = np.subtract(image, self.per_pixel_mean)
        image = np.divide(image, self.per_pixel_std)
        label = self.valid_label[index]
        return image, label

    def read_hessian_image (self, sample_id):
        index = self.train_sample_offset + sample_id.numpy()
        image = self.train_data[index]
        image = np.subtract(image, self.per_pixel_mean)
        image = np.divide(image, self.per_pixel_std)
        label = self.train_label[index]
        return image, label

    def augmentation(self, x, y):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y

    def train_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_local_train_samples))
        dataset = dataset.map(lambda x: tf.py_function(self.read_train_image, inp = [x], Tout = [tf.float32, tf.float32]), num_parallel_calls = AUTOTUNE)
        dataset = dataset.map(self.augmentation, num_parallel_calls = AUTOTUNE)
        dataset = dataset.batch(self.train_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples))
        dataset = dataset.map(lambda x: tf.py_function(self.read_valid_image, inp = [x], Tout = [tf.float32, tf.float32]), num_parallel_calls = AUTOTUNE)
        dataset = dataset.batch(self.valid_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def boundary_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.train_batch_size))
        dataset = dataset.map(lambda x: tf.py_function(self.read_hessian_image, inp = [x], Tout = [tf.float32, tf.float32]), num_parallel_calls = AUTOTUNE)
        dataset = dataset.batch(1)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def hessian_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.train_batch_size))
        dataset = dataset.map(lambda x: tf.py_function(self.read_hessian_image, inp = [x], Tout = [tf.float32, tf.float32]), num_parallel_calls = AUTOTUNE)
        dataset = dataset.batch(self.train_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()
