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
import numpy as np
from typing import Union, Tuple
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.regularizers import l2
from layers import(
    CPConv2D,
    CPLoRAConv2D,
    TuckerConv2D,
    TuckerLoRAConv2D,
    LowRankConv2D,
    LoRAConv2D
)
from typing import Union, Tuple, List
import math
def copy_model (model):
    new_model = tf.keras.models.clone_model(model)
    for i in range (len(model.trainable_variables)):
        new_model.trainable_variables[i].assign(model.trainable_variables[i])
    for i in range (len(model.non_trainable_variables)):
        new_model.non_trainable_variables[i].assign(model.non_trainable_variables[i])
    return new_model

class resnet20_decomposition ():
    def __init__ (self, weight_decay, partial_layers, ranks, decomposition = None):
        self.weight_decay = weight_decay
        self.regularizer = None
        self.initializer = tf.keras.initializers.GlorotUniform(seed = int(time.time()))
        self.partial_layers = partial_layers
        self.ranks = ranks
        self.layer_index = 0
        self.method = decomposition
        print (self.partial_layers)
        print (self.ranks)
        print (self.method)

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        if self.layer_index in self.partial_layers:
            if self.method == "CP":
                x = CPConv2D(num_filters,
                           (3, 3),
                           padding = "same",
                           strides = strides,
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(input_tensor)

            elif self.method == "Tucker":
                x = TuckerConv2D(num_filters,
                           (3, 3),
                           padding = "same",
                           strides = strides,
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(input_tensor)

            elif self.method == "SVD":
                x = LowRankConv2D(num_filters,
                           (3, 3),
                           padding = "same",
                           strides = strides,
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_initializer = self.initializer,
                           kernel_regularizer = self.regularizer)(input_tensor)
        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       use_bias = False,
                       kernel_initializer = self.initializer,
                       kernel_regularizer = self.regularizer)(input_tensor)
        self.layer_index += 1
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LowRankConv2D(num_filters,
                           (3, 3),
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_initializer = self.initializer,                           
                           kernel_regularizer = self.regularizer)(x)    
            elif self.method == "CP":
                x = CPConv2D(num_filters,
                           (3, 3),
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x)
            elif self.method == "Tucker":
                x = TuckerConv2D(num_filters,
                           (3, 3),
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x)

        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       kernel_initializer = self.initializer,
                       kernel_regularizer = self.regularizer)(x)
        self.layer_index += 1
        x = BatchNormalization(gamma_initializer = 'zeros')(x)

        if projection:
            if self.layer_index in self.partial_layers:
                shortcut = Conv2D(int(num_filters * self.ranks[self.layer_index]),
                                  (1, 1),
                                  padding = "same",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(input_tensor)
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  padding = "valid",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(shortcut)
            else:
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  padding = "same",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization()(shortcut)
        elif strides != (1, 1):
            if self.layer_index in self.partial_layers:
                shortcut = Conv2D(int(num_filters * self.ranks[self.layer_index]),
                           (1, 1),
                           strides = strides,
                           padding = "same",
                           use_bias = False,
                           kernel_initializer = self.initializer,
                           kernel_regularizer = self.regularizer)(input_tensor)
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  padding = "valid",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(shortcut)
            else:
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  strides = strides,
                                  padding = "same",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LowRankConv2D(16,
                           (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           use_bias = False,
                           rank = int(16 * self.ranks[self.layer_index]),
                           kernel_initializer = self.initializer,
                           kernel_regularizer = self.regularizer)(x_in)
            elif self.method == "CP":
                x = CPConv2D(16,
                           (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           use_bias = False,
                           rank = int(16 * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x_in)
            elif self.method == "Tucker":
                x = TuckerConv2D(16,
                           (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           use_bias = False,
                           rank = int(16 * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x_in)

        else:
            x = Conv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       name='conv0',
                       padding='same',
                       use_bias=False,
                       kernel_initializer = self.initializer,
                       kernel_regularizer = self.regularizer) (x_in)
        self.layer_index += 1
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (3):
            if i == 0:
                x = self.res_block(x, 16, projection = True)
            else:
                x = self.res_block(x, 16)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 32, strides = (2, 2))
            else:
                x = self.res_block(x, 32)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 64, strides = (2, 2))
            else:
                x = self.res_block(x, 64)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        if self.layer_index in self.partial_layers:
            x = Dense(int(10 * self.ranks[self.layer_index]),
                      kernel_initializer = self.initializer,
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)
            y = Dense(10, activation = 'softmax',
                      kernel_initializer = self.initializer,
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)
        else:
            y = Dense(10, activation = 'softmax', name='fully_connected',
                      kernel_initializer = self.initializer,
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)
        self.layer_index += 1
        return Model(x_in, y, name = "resnet20_decomposition")

class LoRAresnet20_decomposition ():
    def __init__ (self, weight_decay, partial_layers, ranks, method = None):
        self.weight_decay = weight_decay
        self.regularizer = None
        self.initializer = tf.keras.initializers.GlorotUniform(seed = int(time.time()))
        self.partial_layers = partial_layers
        self.ranks = ranks
        self.layer_index = 0
        self.method = method
        print (self.partial_layers)
        print (self.ranks)

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LoRAConv2D(num_filters,
                           (3, 3),
                           strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(input_tensor)
            elif self.method == "Tucker":
                x = TuckerLoRAConv2D(num_filters,
                           (3, 3),
                           strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(input_tensor)
            elif self.method == "CP":
                x = CPLoRAConv2D(num_filters,
                           (3, 3),
                           strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(input_tensor)

        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       use_bias = False,
                       kernel_initializer = self.initializer,
                       kernel_regularizer = self.regularizer)(input_tensor)
        self.layer_index += 1
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LoRAConv2D(num_filters,
                           (3, 3),
                           #strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x)
            elif self.method == "Tucker":
                x = TuckerLoRAConv2D(num_filters,
                           (3, 3),
                           #strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x)
            elif self.method == "CP":
                x = CPLoRAConv2D(num_filters,
                           (3, 3),
                           #strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x)

        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       kernel_initializer = self.initializer,
                       kernel_regularizer = self.regularizer)(x)
        self.layer_index += 1
        x = BatchNormalization(gamma_initializer = 'zeros')(x)

        if projection:
            if self.layer_index in self.partial_layers:
                shortcut = LoRAConv2D(num_filters,
                               (1, 1),
                               #strides = strides,
                               padding = "same",
                               use_bias = False,
                               rank = int(num_filters * self.ranks[self.layer_index]),
                               kernel_regularizer = self.regularizer)(input_tensor)
            else:
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  padding = "same",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization()(shortcut)
        elif strides != (1, 1):
            if self.layer_index in self.partial_layers:
                shortcut = LoRAConv2D(num_filters,
                               (1, 1),
                               strides = strides,
                               padding = "same",
                               use_bias = False,
                               rank = int(num_filters * self.ranks[self.layer_index]),
                               kernel_regularizer = self.regularizer)(input_tensor)
            else:
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  strides = strides,
                                  padding = "same",
                                  use_bias = False,
                                  kernel_initializer = self.initializer,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LoRAConv2D(16,
                           (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           use_bias = False,
                           rank = int(16 * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x_in)
            elif self.method == "Tucker":
                x = TuckerLoRAConv2D(16,
                           (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           use_bias = False,
                           rank = int(16 * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x_in)
            elif self.method == "CP":
                x = CPLoRAConv2D(16,
                           (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           use_bias = False,
                           rank = int(16 * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(x_in)
            
        else:
            x = Conv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       name='conv0',
                       padding='same',
                       use_bias=False,
                       kernel_initializer = self.initializer,
                       kernel_regularizer = self.regularizer) (x_in)
        self.layer_index += 1
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (3):
            if i == 0:
                x = self.res_block(x, 16, projection = True)
            else:
                x = self.res_block(x, 16)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 32, strides = (2, 2))
            else:
                x = self.res_block(x, 32)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 64, strides = (2, 2))
            else:
                x = self.res_block(x, 64)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        if self.layer_index in self.partial_layers:
            x = Dense(int(10 * self.ranks[self.layer_index]),
                      kernel_initializer = self.initializer,
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)
            y = Dense(10, activation = 'softmax',
                      kernel_initializer = self.initializer,
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)
        else:
            y = Dense(10, activation = 'softmax', name='fully_connected',
                      kernel_initializer = self.initializer,
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)
        self.layer_index += 1
        return Model(x_in, y, name = "resnet20_decomposeLoRA")

class wideresnet28_decomposition():
    def __init__ (self, weight_decay, partial_layers, ranks, method = None):
        self.weight_decay = weight_decay
        self.partial_layers = partial_layers
        self.ranks = ranks
        self.layer_index = 0
        self.method = method
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        if self.layer_index in self.partial_layers:
            if self.method == "CP":
                x = CPConv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       rank = int(num_filters*self.ranks[self.layer_index]),
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(input_tensor)
            elif self.method == "SVD":
                x = LowRankConv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       rank = int(num_filters*self.ranks[self.layer_index]),
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(input_tensor)
            elif self.method == "Tucker":
                x = TuckerConv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       rank = int(num_filters*self.ranks[self.layer_index]),
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(input_tensor)
                
        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(input_tensor)
        self.layer_index += 1
        x = BatchNormalization(momentum = self.batch_norm_momentum,  epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = Dropout(0.3)(x)

        if self.layer_index in self.partial_layers:
            if self.method == "CP":
                x = CPConv2D(num_filters,
                        (3, 3),
                        padding = "same",
                        rank = int(num_filters*self.ranks[self.layer_index]),
                        use_bias = False,
                        kernel_regularizer = self.regularizer)(x)
            elif self.method == "SVD":
                x = LowRankConv2D(num_filters,
                        (3, 3),
                        padding = "same",
                        rank = int(num_filters*self.ranks[self.layer_index]),
                        use_bias = False,
                        kernel_regularizer = self.regularizer)(x)
            elif self.method == "Tucker":
                x = TuckerConv2D(num_filters,
                        (3, 3),
                        padding = "same",
                        rank = int(num_filters*self.ranks[self.layer_index]),
                        use_bias = False,
                        kernel_regularizer = self.regularizer)(x)
        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(x)
        self.layer_index += 1
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                                epsilon = self.batch_norm_epsilon)(x)

        if projection:
            shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  padding = "same",
                                  use_bias = False,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                epsilon = self.batch_norm_epsilon)(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  strides = strides,
                                  padding = "same",
                                  use_bias = False,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                epsilon = self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        self.regularizer = l2(self.weight_decay)
        d = 28
        k = 10
        rounds = int((d - 4) / 6)

        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        if self.layer_index in self.partial_layers:
            if self.method == "CP":
                x = CPConv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       rank = int(16*self.ranks[self.layer_index]),
                       padding='same',
                       use_bias=False,
                       kernel_regularizer = self.regularizer) (x_in)
            elif self.method == "SVD":
                x = LowRankConv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       rank = int(16*self.ranks[self.layer_index]),
                       padding='same',
                       use_bias=False,
                       kernel_regularizer = self.regularizer) (x_in)
            if self.method == "Tucker":
                x = TuckerConv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       rank = int(16*self.ranks[self.layer_index]),
                       padding='same',
                       use_bias=False,
                       kernel_regularizer = self.regularizer) (x_in)
 
        else:
            x = Conv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       name='conv0',
                       padding='same',
                       use_bias=False,
                       kernel_regularizer = self.regularizer) (x_in)
        self.layer_index += 1
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 16 * k, projection = True)
            else:
                x = self.res_block(x, 16 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 32 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 32 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 64 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 64 * k)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        if self.layer_index in self.partial_layers:
            x = Dense(int(100 * self.ranks[self.layer_index]),
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)

            y = Dense(100, activation = 'softmax', name='fully_connected',
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)

        else:
            y = Dense(100, activation = 'softmax', name='fully_connected',
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "wideresnet28-10-decomposition")

class LoRAwideresnset28_decomposition():
    def __init__ (self, weight_decay, partial_layers, ranks, method = None):
        self.weight_decay = weight_decay
        self.partial_layers = partial_layers
        self.ranks = ranks
        self.layer_index = 0
        self.method = method
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LoRAConv2D(num_filters,
                        (3, 3),
                        strides = strides,
                        padding = "same",
                        use_bias = False,
                        rank = int(num_filters * self.ranks[self.layer_index]),
                        kernel_regularizer = self.regularizer)(input_tensor)
            elif self.method == "CP":
                x = CPLoRAConv2D(num_filters,
                        (3, 3),
                        strides = strides,
                        padding = "same",
                        use_bias = False,
                        rank = int(num_filters * self.ranks[self.layer_index]),
                        kernel_regularizer = self.regularizer)(input_tensor)
            if self.method == "Tucker":
                x = TuckerLoRAConv2D(num_filters,
                        (3, 3),
                        strides = strides,
                        padding = "same",
                        use_bias = False,
                        rank = int(num_filters * self.ranks[self.layer_index]),
                        kernel_regularizer = self.regularizer)(input_tensor)

        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       strides = strides,
                       padding = "same",
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(input_tensor)
        self.layer_index += 1
        x = BatchNormalization(momentum = self.batch_norm_momentum,  epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = Dropout(0.3)(x)

        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LoRAConv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       rank = int(num_filters * self.ranks[self.layer_index]),
                       kernel_regularizer = self.regularizer)(x)
            elif self.method == "CP":
                x = CPLoRAConv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       rank = int(num_filters * self.ranks[self.layer_index]),
                       kernel_regularizer = self.regularizer)(x)
            elif self.method == "Tucker":
                x = TuckerLoRAConv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       rank = int(num_filters * self.ranks[self.layer_index]),
                       kernel_regularizer = self.regularizer)(x)

        else:
            x = Conv2D(num_filters,
                       (3, 3),
                       padding = "same",
                       use_bias = False,
                       kernel_regularizer = self.regularizer)(x)
        self.layer_index += 1
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                                epsilon = self.batch_norm_epsilon)(x)

        if projection:
            if self.layer_index in self.partial_layers:
                shortcut = LoRAConv2D(num_filters,
                                  (1, 1),
                                  padding = "same",
                                  use_bias = False,
                                  rank = int(num_filters * self.ranks[self.layer_index]),
                                  kernel_regularizer = self.regularizer)(input_tensor)
            else:
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  padding = "same",
                                  use_bias = False,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                epsilon = self.batch_norm_epsilon)(shortcut)

        elif strides != (1, 1):
            if self.layer_index in self.partial_layers:
                shortcut = LoRAConv2D(num_filters,
                           (1, 1),
                           strides = strides,
                           padding = "same",
                           use_bias = False,
                           rank = int(num_filters * self.ranks[self.layer_index]),
                           kernel_regularizer = self.regularizer)(input_tensor)
                shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
            else:
                shortcut = Conv2D(num_filters,
                                  (1, 1),
                                  strides = strides,
                                  padding = "same",
                                  use_bias = False,
                                  kernel_regularizer = self.regularizer)(input_tensor)
            self.layer_index += 1
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                epsilon = self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        self.regularizer = None
        d = 28
        k = 10
        rounds = int((d - 4) / 6)

        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        if self.layer_index in self.partial_layers:
            if self.method == "SVD":
                x = LoRAConv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       padding='same',
                       use_bias=False,
                       rank = int(16 * self.ranks[self.layer_index]),
                       kernel_regularizer = self.regularizer) (x_in)
            elif self.method == "CP":
                x = CPLoRAConv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       padding='same',
                       use_bias=False,
                       rank = int(16 * self.ranks[self.layer_index]),
                       kernel_regularizer = self.regularizer) (x_in)
            elif self.method == "Tucker":
                x = TuckerLoRAConv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       padding='same',
                       use_bias=False,
                       rank = int(16 * self.ranks[self.layer_index]),
                       kernel_regularizer = self.regularizer) (x_in)

        else:
            x = Conv2D(16,
                       (3, 3),
                       strides=(1, 1),
                       name='conv0',
                       padding='same',
                       use_bias=False,
                       kernel_regularizer = self.regularizer) (x_in)
        self.layer_index += 1
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 16 * k, projection = True)
            else:
                x = self.res_block(x, 16 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 32 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 32 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 64 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 64 * k)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        if self.layer_index in self.partial_layers:
            x = Dense(int(100 * self.ranks[self.layer_index]),
                      kernel_regularizer = self.regularizer,
                      bias_regularizer = self.regularizer)(x)

            y = Dense(100, activation = 'softmax', name='fully_connected',
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)

        else:
            y = Dense(100, activation = 'softmax', name='fully_connected',
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "wideresnet28-10_LoRA_Decompostion")