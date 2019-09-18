# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""

This version does not contain the model compression components (
sparsification and quantization).

Original paper: (https://arxiv.org/pdf/1602.07360.pdf)
Original Code: https://github.com/tensorflow/tpu/tree/master/models/official/squeezenet/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modelutil import *

import tensorflow.contrib.slim as slim
def fire_moduleSlim(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs

def squeeze(inputs, num_outputs,isTrainable=False):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze',trainable=isTrainable,biases_initializer=None)

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1',trainable=False,biases_initializer=None)
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3',trainable=False,biases_initializer=None)
    return tf.concat([e1x1, e3x3], 3)



def fire_module(inputs, squeeze_depth,   name,withBias=True,trainable=True,expandRatio=4):
  """Fire module: squeeze input filters, then apply spatial convolutions."""
  expand_depth=squeeze_depth*expandRatio
  with tf.variable_scope(name, "fire", [inputs]):
    squeezed = conv2d(inputs, squeeze_depth, [1, 1],useBias=withBias,isTrainable=trainable, name="squeeze")
    
    e1x1 = conv2d(squeezed, expand_depth, [1, 1],useBias=withBias,isTrainable=trainable, name="e1x1" )
    e3x3 = conv2d(squeezed, expand_depth, [3, 3],useBias=withBias,isTrainable=trainable, name="e3x3" )
    return tf.concat([e1x1, e3x3], axis=3)

#https://github.com/cmasch/squeezenet/blob/master/squeezenet.py
def createModel(images, is_training=True, num_classes=1001,inputResolution=64,weightDir='', defaultDevice='/gpu:0', squeezeNet_11=False,squeezeNetSlim=True, nb_expand_ratio = 4, dropout_rate=None ):
     
    with tf.device(defaultDevice):  
      
      if(squeezeNet_11):  #"""Squeezenet 1.1 model. 2.4x less computation over SqueezeNet 1.0 implemented above."""
          net = conv2d(images, 64, [3, 3], strides=(2, 2), name="conv1")
          net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool1")
          net = fire_module(net, 16,  name="fire2")
          net = fire_module(net, 16,  name="fire3")
          net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool3")
          net = fire_module(net, 32,  name="fire4")
          net = fire_module(net, 32,  name="fire5")
          net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool5")
          net = fire_module(net, 48,  name="fire6")
          net = fire_module(net, 48,  name="fire7")
          net = fire_module(net, 64,  name="fire8")
          net = fire_module(net, 64,  name="fire9")
          if dropout_rate:
              net = tf.layers.dropout(net, rate=dropout_rate if is_training else 0.0, name="drop9")
          
          net = conv2d(net, num_classes, [1, 1], strides=(1, 1), name="conv10")
         
          if(inputResolution==224):
              net = tf.layers.average_pooling2d(net, pool_size=(13, 13), strides=(1, 1))
          elif(inputResolution==64):
              net = tf.layers.average_pooling2d(net, pool_size=(3, 3), strides=(1, 1))
          logits = tf.layers.flatten(net)
      elif(squeezeNetSlim):
          
            useOriginalSlimNet=True
            if(useOriginalSlimNet):
                with tf.variable_scope('squeezenet', [images], reuse=False):
                    with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=is_training):
                        net = slim.conv2d(images, 96, [7, 7], stride=2, scope='conv1',biases_initializer=None)
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
                        net = fire_moduleSlim(net, 16, 64, scope='fire2')
                        net = fire_moduleSlim(net, 16, 64, scope='fire3')
                        net = fire_moduleSlim(net, 32, 128, scope='fire4')
                        net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
                        net = fire_moduleSlim(net, 32, 128, scope='fire5')
                        net = fire_moduleSlim(net, 48, 192, scope='fire6')
                        net = fire_moduleSlim(net, 48, 192, scope='fire7')
                        net = fire_moduleSlim(net, 64, 256, scope='fire8')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
                        net = fire_moduleSlim(net, 64, 256, scope='fire9')
                if dropout_rate:
                  net = tf.layers.dropout(net, rate=dropout_rate if is_training else 0.0, name="drop9")
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
                logits = tf.squeeze(net, [1, 2], name='logits')
            else:
                with tf.variable_scope('squeezenet', [images], reuse=False):
                      net = conv2d(images, 96, [7, 7], strides=(2, 2), name='conv1',useBias=False,isTrainable=False)
                      net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool1" )
                      net = fire_module(net, 16,   name="fire2",withBias=False,trainable=False)
                      net = fire_module(net, 16,   name="fire3",withBias=False,trainable=False)
                      net = fire_module(net, 32,   name="fire4",withBias=False,trainable=False)
                      net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool4" )
                      net = fire_module(net, 32,   name="fire5",withBias=False,trainable=False)
                      net = fire_module(net, 48,   name="fire6",withBias=False,trainable=False)
                      net = fire_module(net, 48,   name="fire7",withBias=False,trainable=False)
                      net = fire_module(net, 64,   name="fire8",withBias=False,trainable=False)
                      net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool8" )
                net = fire_module(net, 64,   name="fire9",withBias=False,trainable=True)
                if dropout_rate:
                  net = tf.layers.dropout(net, rate=dropout_rate if is_training else 0.0, name="drop9")
                net = conv2d(net, num_classes, [1, 1], name='conv10' )   
                net = tf.layers.average_pooling2d(net,pool_size= net.get_shape()[1:3], strides=(1, 1), name='avgpool10')
                logits = tf.squeeze(net, [1, 2], name='logits')
                
      else: #"""Squeezenet 1.0 model."""   
          
          net = conv2d(images, 96, [7, 7], strides=(2, 2), name="conv1")
          net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool1")
          net = fire_module(net, 16,   name="fire2")
          net = fire_module(net, 16,   name="fire3")
          net = fire_module(net, 32,   name="fire4")
          net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool4")
          net = fire_module(net, 32,   name="fire5")
          net = fire_module(net, 48,   name="fire6")
          net = fire_module(net, 48,   name="fire7")
          net = fire_module(net, 64,   name="fire8")
          net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool8")
          net = fire_module(net, 64,   name="fire9")
          if dropout_rate:
              net = tf.layers.dropout(net, rate=dropout_rate if is_training else 0.0, name="drop9")
          
          net = conv2d(net, num_classes, [1, 1], strides=(1, 1), name="conv10")
         
          if(inputResolution==224):
              net = tf.layers.average_pooling2d(net, pool_size=(13, 13), strides=(1, 1))
          elif(inputResolution==64):
              net = tf.layers.average_pooling2d(net, pool_size=(3, 3), strides=(1, 1))
          logits = tf.layers.flatten(net)
    if(len(weightDir)>0):
        tf.train.init_from_checkpoint(weightDir,{"squeezenet/":"squeezenet/"}) 
        print('Load pretrain model from %s as start point.'%weightDir)
    return logits

 