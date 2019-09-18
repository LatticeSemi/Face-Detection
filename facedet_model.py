# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
"""
Facedet design demo:
    
"""
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from modelutil import *

 
def CBRP_module(inputs,  depth,  name,  withPool,is_training=True ):
  """CBRP Subgraph Block:  Conv2D  BatchNorm Relu Pool"""
  with tf.variable_scope(name, "ConvBatchPool", [inputs], reuse=tf.AUTO_REUSE):
    conv = conv2d(inputs,  depth,     name="conv")
    epsBN=1e-3
    meanVarBetaGamma=tf.constant(np.full((4,depth),epsBN,dtype=np.float32), dtype=tf.float32  ) 
    bn = tf.nn.batch_normalization(conv, meanVarBetaGamma[0], meanVarBetaGamma[1], meanVarBetaGamma[2], meanVarBetaGamma[3],  epsBN,name='batchnorm'  )
    relu = tf.nn.relu(bn , name="relu" )
    if(withPool):
        maxPool=tf.nn.max_pool(relu,  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name="maxPooling" )
        return maxPool
    else:
        return relu 


def createModel(images, is_training=True, num_classes=2,inputResolution=32 ):
  """Face detect model."""
  net = CBRP_module    (images, 64 ,  "CBR1",True,is_training)
  net = CBRP_module    (net,    64 , "CBR2",True,is_training)
  net = CBRP_module   (net,    128 , "CBR3",False,is_training)  
  net =  tf.layers.flatten(net,name='Flatten')
  net =  tf.layers.dense(net,num_classes,name='FC', reuse= tf.AUTO_REUSE) 
  return net,None


