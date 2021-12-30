import numpy as np
from ENV.rebort_arm import game_env
import yaml
import os 
import tensorflow as tf 
from tensorflow.python.platform import gfile
import argparse
from utils import inference


def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path', type=str,
                        default='./DDPG/agent.ckpt-3243-55.03.pb', help='save model path')
    parser.add_argument('--input_node_name', type=str,
                        default='state_', help='save model path')
    parser.add_argument('--output_node_name', type=str,
                        default='target_actor/fully_connected_3/Tanh', help='save model path')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = test_args()
    pb_path = args.pb_path
    input_node_name = args.input_node_name
    output_node_name = args.output_node_name
    env_config_path = 'config/rebort_arm_config.yaml'
    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_config = yaml.load(f)
    env = game_env(env_config)
    var = 0.05
    state = env.reset()
    reward_sum = 0


    while True:
        env.render()
        actions = inference(state, pb_path, input_node_name, output_node_name)[0]
        actions = np.clip(np.random.normal(actions, var), -1, 1)
        state_1, reward, done = env.step(actions)
        reward_sum += reward
        state = state_1
        if done:
            state = env.reset()

    





