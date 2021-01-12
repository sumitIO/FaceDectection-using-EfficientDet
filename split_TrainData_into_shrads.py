# %%time

import tensorflow as tf

# filepath to original train.tfrecord
raw_dataset = tf.data.TFRecordDataset("/content/data/train.record")

shards = 10

%mkdir /content/trainingData/
%cd /content/trainingData/

for i in range(shards):
    writer = tf.data.experimental.TFRecordWriter(f'out.tfrecord.{i:03d}')
    writer.write(raw_dataset.shard(shards, i))