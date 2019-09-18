# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:36:23 2019
Abstract Block can be shared in different models
@author: ytan
"""

import tensorflow as tf

def conv2d(inputs,
           filters,
           kernel_size=[3,3],
           strides=(1, 1),
           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
           useBias=True,
           isTrainable=True,
           name=None):
  return tf.layers.conv2d(
      inputs,
      filters,
      kernel_size,
      strides,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      activation=tf.nn.relu,
      use_bias=useBias,
      trainable=isTrainable,
      name=name,
      padding="same")
  
  
  
  
