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
import tensorflow as tf
import numpy as np

'''
List cosine similarity.
Each input is a list of tensors.
It returns the cosine similarity value calculated from the whole tensors of the two lists.
'''
def cos_sim (a, b):
    sum_a = 0
    sum_b = 0
    dot = 0
    for i in range (len(a)):
        sum_a += tf.norm(a[i])**2
        sum_b += tf.norm(a[i])**2
        dot += tf.reduce_sum(tf.math.multiply(a[i], b[i]))
    norm_a = tf.math.sqrt(sum_a)
    norm_b = tf.math.sqrt(sum_b)
    sim = dot / (norm_a * norm_b)
    return sim
