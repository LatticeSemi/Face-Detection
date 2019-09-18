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
"""Imagenet loading and preprocessing pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
#Dependency only needed for debug
from PIL import Image
import numpy as np
import sys  
from enum import Enum     # for enum34, or the stdlib version
rootOfDataFolder=r'./Data/' 

LEARNRATE=0.001  


ISTRAINING=True

evaluateTrainingData=False  #When do evaluation if True will evaluate training data for debug purpose.  Should normally set to false as most time we need evaluate valiation dataset


evaluateTypeIndex=1    

# from aenum import Enum  # for the aenum version
 
DATAENUM = Enum('DATAENUM', ["train","validation","test"])
evaluateDataType=DATAENUM.test
     


#######################Advanced Parameters which are usually stay unchanged for training ############################################## 
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale 
IMAGERES=64  #Need to <= than all image ,  catdog\val\cat\cat.1000   is 150x149
normalizeImageFactor=255.0
MAXBATCH=16
if(ISTRAINING):
    DROP_BATCH_REMAINDER  =True  #For training BN of small set of remainder is not good for training
else:#Want to use all validate   or test dataset
    DROP_BATCH_REMAINDER  =False

imgValTrainNumFileName='imgValTrainNum.txt'
labels_file=rootOfDataFolder+'label.txt'

EPOCHRUN=2000000 
EVALUATEGAPINSECOND=1800

MAX_NUM_CKPT=3
PARALLELNUMER=4
PREFETCHDATA=1
ITERATION_PER_LOOP=1
TRAINMOMENT=0.9
TPU_SHARD_NUM=1
substractImageNetMean=False
MIN_LEARNINGRATE=0.01*LEARNRATE
LOG_ITERATION_GAP=100
# Image Parameters
WITHDUMMYCLASS=False
if(substractImageNetMean):
    _R_MEAN = 123.68 / normalizeImageFactor
    _G_MEAN = 116.78 / normalizeImageFactor
    _B_MEAN = 103.94 / normalizeImageFactor
else: 
    _R_MEAN = 0
    _G_MEAN = 0
    _B_MEAN = 0
USE_TPU=False

tensorRecordFolderName=r"./tfrecord/train/"
_FILE_PATTERN = "%s-*" 
_SPLITS = set([DATAENUM.train.name,DATAENUM.validation.name,DATAENUM.test.name,]) 
if(os.path.isfile(labels_file)):
      unique_labels = [l.strip() for l in tf.gfile.FastGFile(
          labels_file, 'r').readlines()]     
else:
      print("Required file (%s)  don't exists.  "% labels_file)
      sys.exit()

textNames=[]
if(WITHDUMMYCLASS):
    N_CLASSES =len(unique_labels)+1
    textNames.append['background']
    for i in range( len(unique_labels)):
        textNames.append(unique_labels[i])
     # Also need to update  
     # WITHDUMMYCLASS need to be consistent in T:\FacedetDemo\src\build_image_data.py
else:
    N_CLASSES = len(unique_labels)# CHANGE HERE, total number of classes
    textNames=unique_labels
    
IMG_HEIGHT = IMAGERES # CHANGE HERE, the image height to be resized to
IMG_WIDTH = IMG_HEIGHT # CHANGE HERE, the image width to be resized to
INPUTIMAGERESOLUTION=IMG_HEIGHT #Assume square input

_RESIZE_SIDE_MIN = IMG_HEIGHT
_RESIZE_SIDE_MAX = IMG_HEIGHT


 
imgNumFile=rootOfDataFolder+tensorRecordFolderName+imgValTrainNumFileName

if(os.path.isfile(imgNumFile)):
    numValTrain=np.loadtxt(imgNumFile)
    NUMBEROfTest=int(numValTrain[0])
    NUMBERPEREVAL=int(numValTrain[1])
    NUMBERPEREPOCH=int(numValTrain[2])
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&& IMAGERES=%d   Test case=%d validation case=%d training case=%d  at %s  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'%(IMAGERES,NUMBEROfTest,NUMBERPEREVAL,NUMBERPEREPOCH,imgNumFile))
else:   
    print('Can not find training number information file: %s. Please create one by running build_image_data.py'%imgNumFile) 
    NUMBERPEREVAL=MAXBATCH
    NUMBERPEREPOCH=MAXBATCH*10

NUMBER_OF_EVALUATION=1  #Not in use
BATCHSIZE=int(min(MAXBATCH,NUMBERPEREVAL )) 
 
BUFFER_SIZE_UNIT=min(BATCHSIZE*10, NUMBERPEREPOCH) 
LEARNRATESCALE=1 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn"t assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), CHANNELS), ["Rank of image must be equal to 3."])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ["Crop size greater than the image size."])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError("Empty image_list.")

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, CHANNELS), [
            "Wrong rank for tensor  %s [expected] [actual]", image_list[i].name,
            CHANNELS, image_rank
        ])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ["Crop size greater than the image size."])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height), [
            "Wrong height for tensor %s [expected][actual]", image.name, height,
            image_height
        ])
    width_assert = tf.Assert(
        tf.equal(width, image_width), [
            "Wrong width for tensor %s [expected][actual]", image.name, width,
            image_width
        ])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

  return [
      _crop(image, offset_height, offset_width, crop_height, crop_width)
      for image in image_list
  ]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(
        _crop(image, offset_height, offset_width, crop_height, crop_width))
  return outputs


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn"t match the
      number of values in `means`.
  """
  if image.get_shape().ndims != CHANNELS:
    raise ValueError("Input must be of size [height, width, C>0]")
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError("len(means) must match the number of channels")

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(
      tf.greater(height, width), lambda: smallest_side / width,
      lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(
      image, [new_height, new_width], align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """
  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)
  image = _aspect_preserving_resize(image, resize_side)
  image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, CHANNELS])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, output_height, output_width, resize_side):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side)
  image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width,CHANNELS])
  image = tf.to_float(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image,
                     output_height,
                     output_width,
                     is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].
  Returns:
    A preprocessed image.
  """
  if is_training:
    image = preprocess_for_train(image, output_height, output_width,
                                 resize_side_min, resize_side_max)
  else:
    image = preprocess_for_eval(image, output_height, output_width,
                                resize_side_min)
  if(substractImageNetMean): 
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
  return image


def get_split(split_name, dataset_dir):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS:
    raise ValueError("split name %s was not recognized." % split_name)

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
  return tf.data.Dataset.list_files(file_pattern, shuffle=False)


def fetchImgInforFromRecords(tfrecords_filename):  
    reconstructed_images = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    
    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        height = int(example.features.feature['image/height']
                                     .int64_list
                                     .value[0])
        
        width = int(example.features.feature['image/width']
                                    .int64_list
                                    .value[0])
        
        img_string = (example.features.feature['image/encoded']
                                      .bytes_list
                                      .value[0])
        
        labelIndex= (example.features.feature['image/class/label']
                                    .int64_list
                                    .value[0])
        
        annotation_string =  (example.features.feature['image/class/text']
                                    .bytes_list
                                    .value[0]).decode('utf-8')
        fileName = (example.features.feature['image/filename']
                                    .bytes_list
                                    .value[0]).decode('utf-8')
        with tf.Session() as sess:
            reconstructed_img = tf.image.decode_jpeg(img_string).eval().reshape((height, width, -1))
        
         
        
        reconstructed_images.append((reconstructed_img,labelIndex, annotation_string,fileName))
    return reconstructed_images


class InputReader(object):
  """Provides TFEstimator input function for imagenet, with preprocessing."""

  def __init__(self, data_dir, dataType,  image_width=IMG_HEIGHT, image_height=IMG_WIDTH):
    self._is_training = (DATAENUM.train.name==dataType.name)
    self.dataType = dataType
    self._data_dir = data_dir
    self._image_width = image_width
    self._image_height = image_height

  def _int64_feature(value):
      """Wrapper for inserting int64 features into Example proto."""
      if not isinstance(value, list):
        value = [value]
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    
  def _bytes_feature(value):
      """Wrapper for inserting bytes features into Example proto."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  def _parse_record(self, record):
    """Parse an Imagenet record from a tf.Example.""" 
    
    keys_to_features={
      'image/height':tf.FixedLenFeature([],dtype=tf.int64)  ,
      'image/width': tf.FixedLenFeature([],dtype=tf.int64),
      'image/colorspace': tf.FixedLenFeature((), tf.string, "")  ,
      'image/channels': tf.FixedLenFeature([],dtype=tf.int64),
      'image/class/label': tf.FixedLenFeature([],dtype=tf.int64),
      'image/class/text': tf.FixedLenFeature([], tf.string, ""),
      'image/format':  tf.FixedLenFeature((), tf.string, ""),
      'image/filename': tf.FixedLenFeature([], tf.string, ""),
      'image/encoded': tf.FixedLenFeature((), tf.string, "")}

    parsed = tf.parse_single_example(record, keys_to_features)
   
    image = tf.image.decode_image(
        tf.reshape(parsed["image/encoded"], shape=[]), CHANNELS)
     
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = preprocess_image(
        image=image,
        output_height=self._image_width,
        output_width=self._image_height,
        is_training=self._is_training)

    label = tf.cast(
        tf.reshape(parsed["image/class/label"], shape=[]), dtype=tf.int32)
    print('parsed["image/class/text"]',parsed["image/class/text"])
#    print('label',label )
#    print('image',image)
    
    
    return image, label
  
  

  def __call__(self, params):
    bs = params["batch_size"]
    print('--------------------self._data_dir',self._data_dir)
    
    
    dataset = get_split(
        split_name=self.dataType.name,
        dataset_dir=self._data_dir)

    if self._is_training:
      dataset = dataset.shuffle(buffer_size=BUFFER_SIZE_UNIT)
      dataset = dataset.repeat()

    def _load_records(filename):
      return tf.data.TFRecordDataset(filename, buffer_size=PARALLELNUMER * BUFFER_SIZE_UNIT * BUFFER_SIZE_UNIT)

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _load_records, sloppy=True, cycle_length=PARALLELNUMER*2))

    dataset = dataset.prefetch(bs * PREFETCHDATA)
    dataset = dataset.map(self._parse_record, num_parallel_calls=PARALLELNUMER)
    dataset = dataset.batch(bs, drop_remainder=DROP_BATCH_REMAINDER)
    dataset = dataset.prefetch(PREFETCHDATA)

    features, labels = dataset.make_one_shot_iterator().get_next()
    labels = tf.cast(labels, tf.int32) 
    features.set_shape([bs, IMG_HEIGHT, IMG_WIDTH, CHANNELS])
     
    labels.set_shape([bs])
    return features, labels


 
