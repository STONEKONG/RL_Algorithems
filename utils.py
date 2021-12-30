import numpy as np 
import random
import os 
import tensorflow as tf 
from tensorflow.python.platform import gfile


class Experience_Buffer():

    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        idx = len(experience) + len(self.buffer) - self.buffer_size
        if idx >= 0:
            self.buffer[0:idx] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(
            np.array(
                random.sample(self.buffer, size)
            ),
            [size, 5]
        )

def inference(image, pb_path, input_node_names, output_node_names):

    image_list = np.expand_dims(image, axis=0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') # 导入计算图
    sess.run(tf.global_variables_initializer())
    input = sess.graph.get_tensor_by_name(input_node_names + ':0')
    output = sess.graph.get_tensor_by_name(output_node_names + ':0')
    return sess.run(output, feed_dict={input:image_list})
   

