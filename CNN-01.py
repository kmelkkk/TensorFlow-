import tensorflow as tf
import os
import tarfile
import requests

# url

url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
model_dir = "inception_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

filename = url.split('/')[-1]
filepath = os.path.join(model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

print("完成： ", filename)

# 解压文件

tarfile.open(filepath, 'r:gz').extractall(model_dir)

# 模型结构存放文件
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 训练好的模型

inception_graph_file=os.path.join(model_dir,'classify_image_graph_def.pb')

with tf.Session() as  sess:
    with tf.gfile.FastGFile(inception_graph_file,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')
    #保存图的结构
    writer=tf.summary.FileWriter(log_dir,sess.graph)
    writer.close()
