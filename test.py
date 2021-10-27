

import numpy as np
from gridworld import game_env
import matplotlib.pyplot as plt 
import os 
import tensorflow as tf 
from tensorflow.python.platform import gfile
import yaml 

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

if __name__ is '__main__':

    pb_path = 'agent.ckpt-227-42.46.pb'
    env_config_path = 'env_config.yaml'
    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_config = yaml.load(f)
    env = game_env(env_config)
    
    state = env.state
    reward_sum = 0
    gold_n = 0
    fire_n = 0
    random = False
    plt.ion()
    while True:
        action_r = np.random.randint(0,4) 
        Q_value = inference(state, pb_path)
        action = np.argmax(Q_value, axis=1)[0]
        state_1, reward, d = env.step(action)
        reward_sum += reward
        if reward > 0:
            gold_n += 1
        elif reward <= -1:
            fire_n -= 1
        state = state_1
        plt.imshow(state_1)
        plt.text(30.0, 5.0, 'Gamma Zhu')
        plt.text(32.0, 79.0, 'reward:{:.2f}'.format(reward_sum))
        plt.text(32.0, 83.0, 'goal:{} fire:{}'.format(gold_n, fire_n))
        plt.pause(0.05)
        plt.clf()
        

