
from keras.models import load_model
import cv2
import numpy as np 
import tensorflow as tf


img_path = './1.jpg'


VOC_COLOR = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]], dtype=np.uint8)

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    model = load_model('/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.h5')

img = cv2.imread(img_path, 1)
img =  cv2.resize(img,(224,224), interpolation =  cv2.INTER_AREA)
img = img.astype(np.float32)
img[:, :, 0] -= 103.939
img[:, :, 1] -= 116.779
img[:, :, 2] -= 123.68
img = img[:, :, ::-1]
img = img.reshape((1,224,224,3))

output = model.predict(img)

output = output.reshape((112,112,5))
pre = np.zeros((112,112,3))


for i in range(112):
    for j in range(112):
        pre[i,j,:] =  VOC_COLOR[np.argmax(a[i,j,:])] 

cv2.imwrite('/pre.jpg',pre)



