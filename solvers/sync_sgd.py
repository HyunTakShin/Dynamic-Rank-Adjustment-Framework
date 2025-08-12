'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/03/23
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/05/30
Hyuntak Shin, BA Participant.
<hyuntakshin@inha.ac.kr>
'''
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

class SyncSGD:
    def __init__ (self, num_classes, models, target_layers, compression_ratios):
        self.num_classes = num_classes
        self.optimizers = []
        for i in range (len(models)):
            self.optimizers.append(SGD(momentum = 0.9))
        self.target_layers = target_layers
        self.compression_ratios = compression_ratios
        self.models = models
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
        print ("Synchronous SGD is the solver!")
        self.scan(models)

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def tucker_reconstruct(self, conv1,core,conv2):
        proj_input = np.reshape(conv1, (conv1.shape[2],conv1.shape[3]))
        proj_output = np.reshape(conv2, (conv2.shape[2],conv2.shape[3]))

        h = core.shape[0]
        w = core.shape[1]
        i = conv1.shape[2]
        o = conv2.shape[-1]

        recover = np.zeros(shape = (h,w,i,o))

        for i in range(h):
            for j in range(w):
                temp = np.matmul(core[i][j], proj_output)
                recover[i][j] = np.matmul(proj_input, temp)

        return recover

    def cp_reconstruct(self, param1, param2, param3, param4):
        input = param1.shape[2]
        output = param4.shape[-1]
        kernel = param3.shape[0]
        print(input, output, kernel)
        param1 = np.reshape(param1, (param1.shape[2],param1.shape[-1]))
        param4 = np.reshape(param4, (param4.shape[2],param4.shape[-1]))
        recover = np.zeros(shape = (kernel, kernel, input, output))
        w23 = np.matmul(param2,param3)
        for i in range(kernel):
            for j in range(kernel):
                temp = np.matmul(param1, w23[i][j])
                recover[i][j] = np.matmul(temp, param4)
        return recover
        
    def train_step (self, checkpoint, data, label, target):
        model = checkpoint.models[target]
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])

        grads = tape.gradient(total_loss, model.trainable_variables)
        checkpoint.optimizers[target].apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id, target):
        model = checkpoint.models[target]
        # trainable parameters
        for i in range (len(model.trainable_variables)):
            local_param = model.trainable_variables[i]

    def partial_rank (self, us, v, compression_ratio):
        new_rank = int(us.shape[-1] * compression_ratio)
        us = us[:, :new_rank]
        v = v[:new_rank, :]
        return us, v

    def reconstruct (self, src_tensor_id, src, dst_tensor_id, dst):
        dot_product = np.dot(src.trainable_variables[src_tensor_id[0]], src.trainable_variables[src_tensor_id[1]])
        return dot_product

    def recover_model (self, checkpoint, src_model_id, dst_model_id):
        # Recover trainable variables first.
        src = checkpoint.models[src_model_id]
        dst = checkpoint.models[dst_model_id]

        num_extra_tensors = 0
        for i in range(len(dst.trainable_variables)):
            if i in self.layer_weights:
                index = np.where(np.array(self.layer_weights) == i)[0][0]
                dst_tensor_id = i
                src_tensor_id = self.partial_layer_weights[index]
                if type(src_tensor_id) == list: # Reconstruct and just copy.
                    dot_product = self.reconstruct(src_tensor_id, src, dst_tensor_id, dst)
                    param = np.reshape(dot_product, dst.trainable_variables[dst_tensor_id].shape)
                    dst.trainable_variables[dst_tensor_id].assign(param)
                    num_extra_tensors += 1
                else: # Just copy
                    dst.trainable_variables[dst_tensor_id].assign(src.trainable_variables[src_tensor_id])
            else:
                dst.trainable_variables[i].assign(src.trainable_variables[i + num_extra_tensors])

        # Then, recover non-trainable variables.
        for i in range (len(dst.non_trainable_variables)):
            dst.non_trainable_variables[i].assign(src.non_trainable_variables[i])

    def recover_tucker_model (self, checkpoint, src_model_id, dst_model_id):
        # Recover trainable variables first.
        src = checkpoint.models[src_model_id]
        dst = checkpoint.models[dst_model_id]

        for i in range(len(src.layers)):
            num_params = len(src.layers[i].get_weights())
            #if tucker
            if num_params == 3: 
                param1 = src.layers[i].trainable_variables[0]
                core = src.layers[i].trainable_variables[1]
                param2 = src.layers[i].trainable_variables[2]
                recover = self.tucker_reconstruct(param1,core,param2)
                dst.layers[i].trainable_variables[0].assign(recover)
            else:
                for j in range(len(dst.layers[i].trainable_variables)):
                    dst.layers[i].trainable_variables[j].assign(src.layers[i].trainable_variables[j])

        # Then, recover non-trainable variables.
        for i in range (len(dst.non_trainable_variables)):
            dst.non_trainable_variables[i].assign(src.non_trainable_variables[i])

    def recover_cp_model (self, checkpoint, src_model_id, dst_model_id):
        # Recover trainable variables first.
        src = checkpoint.models[src_model_id]
        dst = checkpoint.models[dst_model_id]

        for i in range(len(src.layers)):
            if hasattr(src.layers[i], 'cp'):        #if cp
                cp1 = src.layers[i].trainable_variables[0]
                cp2 = src.layers[i].trainable_variables[1]
                cp3 = src.layers[i].trainable_variables[2]
                cp4 = src.layers[i].trainable_variables[3]
                recover = self.cp_reconstruct(cp1,cp2,cp3,cp4)
                dst.layers[i].trainable_variables[0].assign(recover)
            else:
                for j in range(len(dst.layers[i].trainable_variables)):
                    dst.layers[i].trainable_variables[j].assign(src.layers[i].trainable_variables[j])

        # Then, recover non-trainable variables.
        for i in range (len(dst.non_trainable_variables)):
            dst.non_trainable_variables[i].assign(src.non_trainable_variables[i])

    def toLoRA (self, checkpoint, src_model_id, dst_model_id):
        src = checkpoint.models[src_model_id]
        dst = checkpoint.models[dst_model_id]

        for i in range (len(dst.layers)):
            num_params = len(dst.layers[i].get_weights())
            if num_params == 0: # input
                continue
            elif num_params == 3: # LoRA layer
                dst.layers[i].non_trainable_variables[0].assign(src.layers[i].trainable_variables[0])
                dst.layers[i].trainable_variables[-1].assign(tf.zeros(dst.layers[i].trainable_variables[-1].shape))
            else: # others
                for j in range (len(src.layers[i].trainable_variables)):
                    dst.layers[i].trainable_variables[j].assign(src.layers[i].trainable_variables[j])
                for j in range (len(src.layers[i].non_trainable_variables)):
                    dst.layers[i].non_trainable_variables[j].assign(src.layers[i].non_trainable_variables[j])

    def toTuckerLoRA (self, checkpoint, src_model_id, dst_model_id):
        src = checkpoint.models[src_model_id]
        dst = checkpoint.models[dst_model_id]

        for i in range (len(dst.layers)):
            if hasattr(dst.layers[i], 'tucker_lora'): # LoRA layer
                dst.layers[i].non_trainable_variables[0].assign(src.layers[i].trainable_variables[0])
                print(dst.layers[i].trainable_variables[1].shape)
                dst.layers[i].trainable_variables[1].assign(tf.zeros(dst.layers[i].trainable_variables[1].shape))
            else: # others
                for j in range (len(src.layers[i].trainable_variables)):
                    dst.layers[i].trainable_variables[j].assign(src.layers[i].trainable_variables[j])
                for j in range (len(src.layers[i].non_trainable_variables)):
                    dst.layers[i].non_trainable_variables[j].assign(src.layers[i].non_trainable_variables[j])

    def toCPLoRA (self, checkpoint, src_model_id, dst_model_id):
        src = checkpoint.models[src_model_id]
        dst = checkpoint.models[dst_model_id]

        for i in range (len(dst.layers)):
            if hasattr(dst.layers[i], 'cp_lora'): # LoRA layer
                dst.layers[i].non_trainable_variables[0].assign(src.layers[i].trainable_variables[0])
                dst.layers[i].trainable_variables[-1].assign(tf.zeros(dst.layers[i].trainable_variables[-1].shape))
            else: # others
                for j in range (len(src.layers[i].trainable_variables)):
                    dst.layers[i].trainable_variables[j].assign(src.layers[i].trainable_variables[j])
                for j in range (len(src.layers[i].non_trainable_variables)):
                    dst.layers[i].non_trainable_variables[j].assign(src.layers[i].non_trainable_variables[j])

    def scan (self, models):
        # List up tensors for all individual layers.
        self.layer_weights = []
        self.partial_layer_weights = []
        offset = 0
        layer_id = 0
        for i in range (len(models[0].trainable_variables)):
            param = models[0].trainable_variables[i]
            if len(param.shape) > 1:
                self.layer_weights.append(i)
                if layer_id in self.target_layers:
                    self.partial_layer_weights.append([i + offset, i + offset + 1])
                    offset += 1
                else:
                    self.partial_layer_weights.append(i + offset)
                layer_id += 1
