import os
import tensorflow as tf

os.chdir('..')
PATH_TO_TFRECORD_TRAIN  = os.path.join(os.getcwd(),'data','train.record')
raw_dataset = tf.data.TFRecordDataset(PATH_TO_TFRECORD_TRAIN)

shards = 10


if os.path.exists(os.path.join(os.getcwd(),'trainingData')) == False:
    os.mkdir('trainingData')

os.chdir(os.path.join(os.getcwd(),'trainingData'))

for i in range(shards):
    writer = tf.data.experimental.TFRecordWriter(f'out.tfrecord.{i:03d}')
    writer.write(raw_dataset.shard(shards, i))