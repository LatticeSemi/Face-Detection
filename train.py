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
Model training implementation with CPU, GPU, TPU support using estimator to abstract and simplify training so
that we can easily use different models as an input of this training (ISTRAINING=True) and inference (ISTRAINING=False) demo 

Training loop and input pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
#import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf 
from  BestCKPT import BestCheckpointCopier

from  data_pipeline import *

#To use TF  model, modify model define script  facedet_model.py    import model here
import facedet_model as mymodel
#To use Keras  model, modify model define script  models_Keras.py    import model here
#import models_Keras as mymodel  



#import facedet_model as mymodel
#import squeezenet_model as mymodel


#from scipy.misc import imread ,imresize  
 
def del_all_flags(FLAGS):
    flags_dict =[ "data_dir"  , "model_dir" ,"preTrainedmodel_dir","save_checkpoints_secs","num_shards","batch_size",
                 "use_tpu","optimizer","momentum","num_epochs","num_evals","learning_rate","min_learning_rate",
                 "iterations_per_loop","num_examples_per_epoch","num_eval_examples","help", "helpfull", "h","helpshort" ]
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        if(keys in FLAGS):
            FLAGS.__delattr__(keys)
#if("data_dir" in tf.app.flags.FLAGS or "help" in tf.app.flags.FLAGS or  "helpfull" in tf.app.flags.FLAGS or "h" in tf.app.flags.FLAGS):
del_all_flags(tf.flags.FLAGS)


# Model training specific paramenters  

flags.DEFINE_string("data_dir", rootOfDataFolder+tensorRecordFolderName, "Location of training files.")
flags.DEFINE_string("model_dir", r"./CKPT", "Where to store model checkpoints.")
flags.DEFINE_string("preTrainedmodel_dir", r"./PretrainedCKPT", "Where to load pretrained weight as start point of training.")
flags.DEFINE_float("learning_rate", LEARNRATE*LEARNRATESCALE, "Learning rate.")
    
flags.DEFINE_integer("num_evals", NUMBER_OF_EVALUATION,
                     "How many times to run an evaluation during training.")
    
flags.DEFINE_float("min_learning_rate",MIN_LEARNINGRATE*LEARNRATESCALE,"The minimal end learning rate.")
flags.DEFINE_integer("save_checkpoints_secs", EVALUATEGAPINSECOND,
                     "Interval between saving model checkpoints.")
flags.DEFINE_integer("batch_size",BATCHSIZE, "Batch size for training and eval.")


flags.DEFINE_string("optimizer", "adam", "Optimizer: momentum|adam|rmsprop")
flags.DEFINE_float("momentum", TRAINMOMENT, "Momentum parameter for SGD optimizer.")
flags.DEFINE_integer("num_epochs",EPOCHRUN,
                     "Number of epochs of the training set to process.")


flags.DEFINE_integer("iterations_per_loop", ITERATION_PER_LOOP,
                     "Number of global step increased per session run.")
flags.DEFINE_integer("num_examples_per_epoch", NUMBERPEREPOCH,
                     "Number of examples to train per epoch.")
flags.DEFINE_integer("num_eval_examples", NUMBERPEREVAL,
                     "Number of examples to evaluate per run.")


# Cloud TPU Cluster Resolvers 
flags.DEFINE_boolean("use_tpu", USE_TPU, "If true, use TPU device.") 
flags.DEFINE_integer("num_shards", TPU_SHARD_NUM, "Number of TPU shards.")

 

FLAGS = flags.FLAGS 
 

def my_model_fn(features, labels, mode, params): 
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  
  logits,model  = mymodel.createModel(
      features, is_training=is_training, num_classes=params["num_classes"],inputResolution=params["image_height"]) #if transfer weight:  weightDir=FLAGS.preTrainedmodel_dir
  
  
  
  
  
   
  if mode == tf.estimator.ModeKeys.PREDICT:
        
        #mymodel.get_layerOutput(model, features) 
         
        return tf.estimator.EstimatorSpec(mode=mode, predictions={
          "classes": tf.argmax(logits, 1),
          "logits":logits,
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      },
        
        
        )
  if(WITHDUMMYCLASS):
      labels=tf.subtract(labels,1)
      
  loss = tf.reduce_mean(
      tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

  global_batch_size = params["num_shards"] * params["batch_size"]
  #decayed_learning_rate = learning_rate *  decay_rate ^ (global_step / decay_steps)
  decayRateConst=int (NUMBERPEREPOCH*0.8)# 1300 * 1000
  decay_steps =decayRateConst * params["num_epochs"] // global_batch_size
  learning_rate = tf.train.polynomial_decay(
      params["lr"],
      global_step=tf.train.get_or_create_global_step(),
      end_learning_rate=params["min_lr"],
      decay_steps=decay_steps,
      power=0.01, #1.0 Linear Decay 
      cycle=False)

 
  lr_repeat = tf.reshape(
      tf.tile(tf.expand_dims(learning_rate, 0), [params["batch_size"],]),
      [params["batch_size"], 1])

  if params["optimizer"] == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  elif params["optimizer"] == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        momentum=params["momentum"],
        epsilon=1.0
    )
  else:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params["momentum"],
        use_nesterov=True)


  predictions={
          "classes": tf.argmax(input=logits, axis=1),
          "logits":logits,
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      } 
  train_op = optimizer.minimize(loss, tf.train.get_global_step())
  accuracy = tf.metrics.accuracy( labels=labels  , predictions=tf.cast( predictions["classes"], tf.int32) ) 
  #To debug, uncomment following hook and #training_hooks=[hook]
  #hook=tf.train.LoggingTensorHook({"output is:": logits, "Labels :": labels},every_n_iter=LOG_ITERATION_GAP)
  #hook=tf.train.LoggingTensorHook({"debug output in test mode:": layer_output_test, "debug output in train mode:": layer_output_train},every_n_iter=LOG_ITERATION_GAP)
  logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
    "accuracy" : accuracy[1]}, every_n_iter=LOG_ITERATION_GAP)

   
  
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops= {
      "validate accuracy":       tf.metrics.accuracy          (labels,  tf.cast(tf.argmax(logits, 1), tf.int32)), 
      "root_mean_squared_error": tf.metrics.root_mean_squared_error (labels,  tf.cast(tf.argmax(logits, 1), tf.int32)),
      "learning_rate":           tf.metrics.mean(learning_rate),
#      "recall":                  tf.metrics.recall            (labels,  tf.cast(tf.argmax(logits, 1), 1)), 
#      "precision":               tf.metrics.precision         (labels,  tf.cast(tf.argmax(logits, 1), 1)), 
#      "precision_at_1":          tf.metrics.precision_at_k    (tf.cast(labels, tf.int64), logits  ,   1), 
#      "recall_at_1":             tf.metrics.recall_at_k       (tf.cast(labels, tf.int64), logits  ,   1),
#      "precision_at_0":          tf.metrics.precision_at_k    (tf.cast(labels, tf.int64), logits  ,   0), 
#      "recall_at_0":             tf.metrics.recall_at_k       (tf.cast(labels, tf.int64), logits  ,   0), 
#      "f1_score":                tf.contrib.metrics.f1_score  (labels ,   tf.cast(tf.argmax(logits, 1), tf.int32) ),
       #Multi class has issue require [0,1] for f1_score_all
       #"f1_score_all":            tf.metrics.mean              (tf.contrib.metrics.f1_score(labels ,   tf.cast(tf.argmax(logits, 1), tf.int32) )),  
     
      } ,
      predictions=predictions,
      #prediction_hooks=[debug_hook],
      training_hooks=[ logging_hook]
  )
 


def main(unused_argv):
  
  #tf.InteractiveSession() 
  tfRecorderFileFolder=FLAGS.data_dir
  if os.path.exists(tfRecorderFileFolder) and os.path.isdir(tfRecorderFileFolder):
        hasTFRecord=False
        for file in os.listdir(tfRecorderFileFolder):
            if file.lower().endswith("00000-of-00001")  :
                hasTFRecord=True
                break
        if not hasTFRecord:
            print("Directory is empty,befor run this script to train, please run build_image_data.py  to create tfrecord  files inside folder: %s" % tfRecorderFileFolder)
            sys.exit()
        else:    
            print('Determining list of input files and labels from folder: %s.' % tfRecorderFileFolder)
  else: 
    os.makedirs(tfRecorderFileFolder, exist_ok=True)
    print("Required Directory don't exists. befor run this script to train, please run build_image_data.py  to create tfrecord  files inside created folder: %s"% tfRecorderFileFolder)
    sys.exit()
    
  tf.reset_default_graph()
  training_examples = FLAGS.num_examples_per_epoch * FLAGS.num_epochs
  eval_examples = FLAGS.num_eval_examples

  params = {
      "num_classes": N_CLASSES,
      "lr": FLAGS.learning_rate,
      "min_lr": FLAGS.min_learning_rate,
      "momentum": FLAGS.momentum,
      "optimizer": FLAGS.optimizer,
      "num_eval_examples": eval_examples,
      "num_shards": FLAGS.num_shards,
      "num_epochs": FLAGS.num_epochs,
      "batch_size":FLAGS.batch_size,
      "image_height":IMG_HEIGHT,
      "eval_delay":EVALUATEGAPINSECOND
  }
  
 

  run_config = tf.estimator.RunConfig(save_summary_steps = LOG_ITERATION_GAP, 
                                      save_checkpoints_steps=LOG_ITERATION_GAP,
                                    keep_checkpoint_max =MAX_NUM_CKPT)
   
  
  estimator = tf.estimator.Estimator(
          config =run_config,
          model_dir = FLAGS.model_dir, 
      model_fn=my_model_fn, 
      params=dict(params, use_tpu=FLAGS.use_tpu),
  )
  

  num_evals = max(FLAGS.num_evals, 1)
  examples_per_eval = training_examples // num_evals
  
  my_input_fn_Train    = InputReader(FLAGS.data_dir, DATAENUM.train)
  my_input_fn_Evaluate = InputReader(FLAGS.data_dir, DATAENUM.validation)
  my_input_fn_test     = InputReader(FLAGS.data_dir, DATAENUM.test)
   
  best_copier = BestCheckpointCopier(
   name='best', # directory within model directory to copy checkpoints to
   checkpoints_to_keep=MAX_NUM_CKPT, # number of checkpoints to keep
   score_metric='validate accuracy', # metric to use to determine "best"
   compare_fn=lambda x,y: x.score > y.score, # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
   sort_key_fn=lambda x: x.score,
   sort_reverse=True) # sort order when discarding excess checkpoints 
  
#  categorical_feature_a = (  tf.feature_column.categorical_column_with_hash_bucket(...))
#  categorical_feature_a_emb = embedding_column( categorical_column=categorical_feature_a)
#  serving_feature_spec = tf.feature_column.make_parse_example_spec(
#      categorical_feature_a_emb)
#  serving_input_receiver_fn = (
#      tf.estimator.export.build_parsing_serving_input_receiver_fn(
#      serving_feature_spec))
#  ckpt_exporter = tf.estimator.BestExporter(
#      name="best_train_ckpt",
#      serving_input_receiver_fn=my_input_fn_Train, 
#      exports_to_keep=MAX_NUM_CKPT)
 
    
#  ckpt_exporter = tf.estimator.BestExporter(
#      compare_fn=_loss_smaller,
#      exports_to_keep=5)
  
#  ckpt_exporter = BestCheckpointCopier(
#  name='best_train', # directory within model directory to copy checkpoints to
#  checkpoints_to_keep=MAX_NUM_CKPT, # number of checkpoints to keep
#  score_metric='validate accuracy', # metric to use to determine "best"
#  compare_fn=lambda x,y: x.score > y.score, # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
#  sort_key_fn=lambda x: x.score,
#  sort_reverse=True) # sort order when discarding excess checkpoints 
  
  
  
  if(ISTRAINING): 
       
      tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=my_input_fn_Train,
                                          max_steps= EPOCHRUN 
                                          ),
        eval_spec=  tf.estimator.EvalSpec(
          name=str('bestTrain'),
          input_fn=my_input_fn_Train if evaluateTrainingData else my_input_fn_Evaluate,
          steps=NUMBERPEREPOCH if evaluateTrainingData else NUMBERPEREVAL  // FLAGS.batch_size,
          exporters=  best_copier  ,
          throttle_secs=EVALUATEGAPINSECOND), 
        ) 
        
  else: # Test Inference results of  a tensorflow recorder
         
     trainingPhrase=False
     debugKerasLayers=True
     tfRecorderPath   = os.path.join(FLAGS.data_dir, "%s-00000-of-00001"% (evaluateDataType.name)) 
     imgInforList= fetchImgInforFromRecords(tfRecorderPath) 
     
     with tf.Session() as sess:
         if(debugKerasLayers):
            
            from PIL import Image
            evalueOneImage=False
            if(evalueOneImage):
                   
                   normalScale=255.0
                   
                   imagePath=r'T:\P4V\trunk\ml\examples\Facedet_FullWorkFlowAndDocker_Demo\Data\moleVScancer\validation\mel\ISIC_0034092.jpg'
                   imagePath=r'T:\P4V\trunk\ml\examples\Facedet_FullWorkFlowAndDocker_Demo\Data\moleVScancer\test\mel\ISIC_0024323.jpg'
                    
                   im=Image.open(imagePath).resize( (IMG_HEIGHT, IMG_WIDTH), Image.LINEAR)
                   
                   rgb_im = im.convert('RGB')
                   r, g, b = rgb_im.getpixel((1, 1))
                   print(r,g,b)
                    #im.show()
                   features=tf.to_float(np.expand_dims(np.array(rgb_im )/normalScale ,0))
                   #weightPath=r'T:\P4V\trunk\ml\examples\Facedet_FullWorkFlowAndDocker_Demo\TrainLog'
                   logits,model  = mymodel.createModel(  features, is_training=trainingPhrase, num_classes=params["num_classes"],inputResolution=params["image_height"] ) #if transfer weight:  weightDir=FLAGS.preTrainedmodel_dir
                   sess.run(tf.global_variables_initializer())     
                   print(' Path=%s logits  value= '%( imagePath ),logits.eval())
                   sys.exit()
            else:
                  imagePath =tfRecorderPath
                  
                  imageList=[] 
                  imageClasse=[]
                  imageFileName=[]
                  for i in range(len(imgInforList)) : 
                    imageList.append(imgInforList[i][0])
                    imageClasse.append(imgInforList[i][2])
                    imageFileName.append(imgInforList[i][3])
                  #features=tf.to_float(np.expand_dims(np.array(imageList)[0] ,0))
                  features=tf.to_float( np.array(imageList)   )
                  logits,model  = mymodel.createModel(  features, is_training=trainingPhrase, num_classes=params["num_classes"],inputResolution=params["image_height"] ) #if transfer weight:  weightDir=FLAGS.preTrainedmodel_dir
                  layerIndex=3
                  layerNames=[]
                  lastIndex=-1
                  #This batch check code is only for Keras, for tensorflow model will return None
                  if(model != None) :
                         for layer in model.layers: 
                                   layerNames.append(layer.name) 
                                   lastIndex=lastIndex+1
                         sess.run(tf.global_variables_initializer())         
                         if(lastIndex!=-1):   
                                 print('Last Layer %s'%layerNames[lastIndex])
                                 output=mymodel.get_layerOutput(model, features,layerNames[lastIndex]) 
                                 results=logits.eval()
                                 #print('logit output of %s'%imagePath,results)
                                 #print('layer output of %s'% layerNames[lastIndex], output )
                          #output=mymodel.get_layerOutput(model, features,'relu/Relu') 
                         correctCount=0
                         className=['cat' ,'dog']
                         totalCount=results.shape[0]
                         for i in range(totalCount) : 
                                  if(results[i][0]>results[i][1] and imageClasse[i] == className[0]): 
                                      correctCount=correctCount+1
                                  if(results[i][0]<results[i][1] and imageClasse[i] == className[1]): 
                                      correctCount=correctCount+1
                                  print('Class=%s Path=%s  value=[%f %f]'%(imageClasse[i],imageFileName[i],results[i][0],results[i][1]))
                         print('Accuracy =%f'%(correctCount*1.0/totalCount))
                         sys.exit()
            
         
     with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())         
            if(evaluateDataType.name==DATAENUM.validation.name):
                predictions=estimator.predict(input_fn=my_input_fn_Evaluate)   
            else:
                predictions=estimator.predict(input_fn=my_input_fn_test)  
                
     predicted_textName=[ ( textNames[p['classes']],p["probabilities"][p['classes']] ,p['logits']) for p in predictions ]
     
     correctCount=0
     totalNum=len(features)
     #If in interactive session/eager:    features, trueLabels=my_input_fn_Evaluate.__call__(params)
     print('evaluation totalNum=%d'%totalNum)
     for i in range(totalNum):
         
         pName=predicted_textName[i][0]
         pProb=predicted_textName[i][1]
         pLog=predicted_textName[i][2]
         
         
         inputImage=imgInforList[i][0]
             
         inputClassText=imgInforList[i][2] 
         inputFileName=imgInforList[i][3] 
         
         if(inputClassText==pName):
             correctCount=correctCount+1
         else:
             print('Incorrect predict of %s (predict %s ground truth %s) , %f  output: '%(inputFileName,pName, inputClassText ,pProb),pLog)
             saveIncorrectImageTotfRecorderFolder=True
             if(saveIncorrectImageTotfRecorderFolder):  
                enc = tf.image.encode_jpeg(inputImage)
                incorrectFolder=FLAGS.data_dir+'/Incorrect/'
                if(os.path.isdir(incorrectFolder)==False):
                    os.mkdir(incorrectFolder)
                
                fname = tf.constant(incorrectFolder+inputFileName)
                fwrite = tf.write_file(fname, enc) 
                sess = tf.Session()
                sess.run(fwrite)
      
     print('Accuracy of %d predication =%.2f'%(totalNum,correctCount*1.0/totalNum )) 
      
       
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
