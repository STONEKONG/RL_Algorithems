
# from os import terminal_size
# import re
import numpy as np
# from tensorflow.python.ops.math_ops import reduce_all 
from gridworld import game_env
import matplotlib.pyplot as plt 
# import cv2 
import os 
import tensorflow as tf 
# from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

reward_dict = {
        'fire': -2, 
        'goal': 1, 
        'dency': -0.1,
        'penalize': -0.5
    }
env = game_env(partial=False, reward_dict=reward_dict, size=7)
# env = game_env(partial=False, size=7)
# env_r = game_env(partial=False, size=7)
plt.ion()
# state = env.reset()

def inference(image, pb_path):
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
    input_1 = sess.graph.get_tensor_by_name('target_netstate:0')
    output = sess.graph.get_tensor_by_name('target_net/Q_value:0')
    Q_value = sess.run(output, feed_dict={input_1:image_list})
    return Q_value

pb_path = 'agent-9500.pb'

state = env.reset()
# state_r = env_r.reset()
reward_sum = 0
# reward_sum_r = 0
correct = 0
fault = 0
# correct_r = 0
# fault_r = 0
random = False
while True:
    action_r = np.random.randint(0,4) 
    Q_value = inference(state, pb_path)
    action = np.argmax(Q_value, axis=1)[0]
    state_1, reward, d = env.step(action)
    reward_sum += reward
    print(reward)
    if reward > 0:
        correct += 1
    elif reward <= -2:
        fault -= 1
    plt.imshow(state_1)
    plt.text(28.0, 5.0, 'Gamma Zhu')
    plt.text(28.0, 79.0, 'reward:{:.2f}'.format(reward_sum))
    plt.text(24.0, 84.0, 'goal:{} fire:{}'.format(correct, fault))
    plt.pause(0.1)
    plt.clf()
    state = state_1
    print(reward)

