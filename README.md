# mobilenet_segnet-keras

Use to train color images and PNG images.

```

#####
#finetuning model

from keras.models import model_from_json
import cv2
import time
import numpy as np 
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

from keras.utils.generic_utils import CustomObjectScope
import keras

input_height = 224
input_width = 224
input_channels = 3

img_input =  keras.models.Input(shape=(input_height, input_width,input_channels))


def my_argamx(feat, img_input):
    x = feat
    x = keras.layers.Lambda(lambda x : tf.argmax(x, axis= 3,output_type=tf.dtypes.int32))(x)
    model = keras.models.Model(img_input, x)
    return model



def my_finetuning_model():
    with open("/home/zhu/procedure/onnx/FCN-ResNet18-Cityscapes-1024x512/keras_train_ckpt/mobilenet_segnet_no_top.json", "r") as f:
        json_string = f.read()
        with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
            base_model_network = model_from_json(json_string)
    keras.models.Model(base_model_network.input, base_model_network.output).load_weights('/home/zhu/procedure/onnx/FCN-ResNet18-Cityscapes-1024x512/keras_train_ckpt/mobilenet_segnet_no_top.h5')
    return base_model_network.input, base_model_network.output

img_input, feat = my_finetuning_model()
model = my_argamx(feat, img_input)
for i in model.layers:
    i.trainable = False

model.save('/home/zhu/procedure/onnx/FCN-ResNet18-Cityscapes-1024x512/keras_train_ckpt/add_argmax_layers.h5')

#####################
#load model

from keras.models import load_model
import cv2
import time
import numpy as np 
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

from keras.utils.generic_utils import CustomObjectScope
import keras


with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6, 'tf':tf}):
    model = load_model('/home/zhu/procedure/onnx/FCN-ResNet18-Cityscapes-1024x512/keras_train_ckpt/add_argmax_layers.h5')

model.summary()
########################
```
