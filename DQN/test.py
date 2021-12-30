

import numpy as np
from ENV.gridworld import game_env
import matplotlib.pyplot as plt 
import tensorflow as tf 
import yaml 
import argparse
from utils import inference


def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path', type=str,
                        default='./DQN/agent.ckpt-452-1.85.pb', help='save model path')
    parser.add_argument('--input_node_name', type=str,
                        default='target_net_state', help='save model path')
    parser.add_argument('--output_node_name', type=str,
                        default='target_net/Q_value', help='save model path')
    return parser.parse_args()

if __name__ is '__main__':

    args = test_args()
    pb_path = args.pb_path
    input_node_name = args.input_node_name
    output_node_name = args.output_node_name
    env_config_path = 'config/gridworld_config.yaml'
    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_config = yaml.load(f)
    env = game_env(env_config)
    epsilon = 0.15
    state = env.state
    reward_sum = 0
    gold_n = 0
    fire_n = 0
    
    plt.ion()
    while True:
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0,4)
        else:
            Q_value = inference(state, pb_path,input_node_name, output_node_name)
            action = np.argmax(Q_value, axis=1)[0]
        # 0 - up, 1 - down, 2 - left, 3 - right
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
        

