
import keras
import tensorflow as tf

import os
import cv2
import random
import itertools
import numpy as np
from keras import backend as K

from keras.layers import Lambda
keras.backend.set_image_data_format('channels_last')



DESIRED_ACCURACY = 0.95

n_classes = 5 
input_height = 224
input_width = 224
input_channels = 3
batch_size = 1


def relu6(x):
    return K.relu(x, max_value=6)

class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')> DESIRED_ACCURACY):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()



img_input =  keras.models.Input(shape=(input_height, input_width,input_channels)) #channels_last
#Conv1
x = keras.layers.Conv2D(32, (3,3), activation='relu',strides=(2, 2),padding='same',use_bias=False)(img_input)
#Conv2-3
x = keras.layers.DepthwiseConv2D( kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(relu6)(x)
x = keras.layers.Conv2D(64, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
f1 = x


#Conv4-5
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
#Conv6-7
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
f2 = x

#Conv8-9
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(256, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
#Conv10-11
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(256, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
f3 = x

#Conv12-13
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
#Conv13-22
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)

f4 = x

#segnet
x = keras.layers.Conv2D(512, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.UpSampling2D((2, 2))(x)

x = keras.layers.Conv2D(256, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.UpSampling2D((2, 2))(x)

x = keras.layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)


x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)


x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Conv2D(n_classes, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)


o_shape = keras.models.Model(img_input, x).output_shape
i_shape = keras.models.Model(img_input, x).input_shape
output_height = o_shape[1]
output_width = o_shape[2]
input_height = i_shape[1]
input_width = i_shape[2]
n_classes = o_shape[3]
x = (keras.layers.Reshape((output_height*output_width,-1)))(x)
#x = (keras.layers.Permute((2, 1)))(x)


x = keras.layers.Activation('softmax')(x)

model = keras.models.Model(img_input, x)

model.output_width = output_width
model.output_height = output_height
model.n_classes = n_classes
model.input_height = input_height
model.input_width = input_width

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['acc']
              )

model.summary()







class DataLoaderError(Exception):
    pass

def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """
    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]
    image_files = []
    segmentation_files = {}
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))
    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))
            segmentation_files[file_name] = (file_extension, full_dir_entry)
    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path,
                                segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))
    return return_value


def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_last'):
    """ Load image array from input """
    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))
    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0
    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img




def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False):
    """ Load segmentation array from input """
    seg_labels = np.zeros((height, width, nClasses))
    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]
    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)
    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))
    return seg_labels




def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False,
                                 augmentation_name="aug_all"):
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)
            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                               augmentation_name)
            X.append(get_image_array(im, input_width,
                                     input_height, ordering= 'channels_last'))
            Y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))
        yield np.array(X), np.array(Y)

train_generator = image_segmentation_generator(images_path  ='/home/nvidia/procedure/keras/JPEGImages/', segs_path = '/home/nvidia/procedure/keras/SegmentationClassAug/',  batch_size = batch_size,  n_classes = n_classes, input_height = input_height, input_width = input_width, output_height = input_height, output_width = input_width)


history = model.fit_generator(
       train_generator,
       steps_per_epoch=2,  
       epochs=2,
       verbose=1,
       callbacks=[callbacks])


model.save('/home/nvidia/procedure/keras/output/mobilenetv1_segnet.h5')
    







