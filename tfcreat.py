import os
import tensorflow as tf
from PIL import Image
import sys

print("please enter accrording to the following format: python tfcreat.py 'imgpath' packnum 'savepath' clustersNum")

if sys.argv[1] and sys.argv[2] and sys.argv[3] and sys.argv[4]:
    cwd = str(sys.argv[1])
    bestnum = int(sys.argv[2])
    filepath = str(sys.argv[3])
    cNum = int(sys.argv[4])-2
else:
    cwd = 'D:/intel match/'
    filepath = 'D:/intel match/'
    bestnum = 800
    cNum = 198

classes=['1','2']
for i in range(cNum):
    classes.append(str(i+3))

num = 0
recordfilenum = 0

tfrecordfilename = ("incs_%.3d.tfrecord" % recordfilenum)
writer = tf.python_io.TFRecordWriter(filepath+tfrecordfilename)

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for index,name in enumerate(classes):
    class_path = cwd+name+'\\'
    for img_name in os.listdir(class_path):
        num = num+1
        if num>bestnum:
            num = 1
            recordfilenum = recordfilenum+1
            tfrecordfilename = ("incs_%.3d.tfrecord" % recordfilenum)
            writer = tf.python_io.TFRecordWriter(filepath+tfrecordfilename)
        
        image_format = b'jpeg'
        height = 224
        width = 224
        img_path = class_path+img_name
        image = tf.gfile.FastGFile(img_path,'rb').read()
        fu={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            }
        example = tf.train.Example(features=tf.train.Features(feature=fu))
        writer.write(example.SerializeToString())

writer.close()