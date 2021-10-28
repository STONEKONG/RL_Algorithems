from __future__ import division 
import numpy as np 
import random
import tensorflow as tf 
import tensorflow.contrib.slim as slim  
import argparse
from gridworld import game_env
import os 
import yaml 


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

class main():
    def __init__(self, args):
        self.args = args
        self.load_env()

    def load_env(self):
        with open(self.args.env_config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
        self.env = game_env(config)

    def cnn(self, input_img, name):

        with tf.variable_scope(name):
            conv_1 = slim.conv2d(input_img, 32, [8,8], [4,4], padding='VALID')
            conv_2 = slim.conv2d(conv_1, 64, [4,4], [2,2], padding='VALID')
            conv_3 = slim.conv2d(conv_2, 64, [3,3], [1,1], padding='VALID')
            conv_4 = slim.conv2d(conv_3, self.args.h_size, [7,7], [1,1], padding='VALID')
            stream_a, stream_v = tf.split(conv_4, 2, 3)
            stream_a = slim.flatten(stream_a)
            stream_v = slim.flatten(stream_v)
            advantage = slim.fully_connected(stream_a, self.env.actions, activation_fn=None)
            value = slim.fully_connected(stream_v, 1, activation_fn=None)
            Q_value = tf.add(value, tf.subtract(
                advantage, tf.reduce_mean(
                    advantage, axis=1, keep_dims=True
                )
            ), name='Q_value')
            return Q_value

    def model(self, name):

        state = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32, name=name+'state')
        Q_value = self.cnn(state, name)
        predict = tf.argmax(Q_value, 1)
        return {
            'state': state, 
            'Q_value': Q_value,
            'predict': predict
        }

    def update_target_graph(self, target_vars, online_vars):
        op_holder = []
        i = 0
        for onl_var, tar_var in zip(online_vars, target_vars):
        # for i, onl_var in enumerate(online_vars):
            op_holder.append(
                target_vars[i].assign(
                    onl_var.value() * self.args.tau + tar_var.value() * (1-self.args.tau)
                )
            )
            i += 1
        return op_holder

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        
        for idx,var in enumerate(tfVars[0:total_vars//2]):

            (var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())
            op_holder.append(tfVars[idx+total_vars//2].assign())
        return op_holder

    def update_target(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def train(self):
        args = self.args
        
        tf.reset_default_graph()
        online_net = self.model('online_net')
        target_net = self.model('target_net')

        target_Q_h = tf.placeholder(shape=[None], dtype=tf.float32)
        actions_h = tf.placeholder(shape=[None], dtype=tf.int32)

        onehot_actions = tf.one_hot(actions_h, self.env.actions, dtype=tf.float32)

        online_Q = tf.reduce_sum(tf.multiply(online_net['Q_value'], onehot_actions), axis=1)
        loss = tf.reduce_mean(tf.square(target_Q_h - online_Q))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.l_r)
        update_model = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online_net')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_net')
        target_Ops = self.update_target_graph(target_vars, online_vars)

        epsilon = args.max_epsilon
        step_drop = (args.max_epsilon - args.min_epsilon) / args.decay_step
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        total_step = 0
        exp_buffer = Experience_Buffer(args.buffer_size)
        r_list = []
        best_mean_reward = 0
        with tf.Session(config=config) as sess:
            sess.run(init)
            if args.load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(args.save_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            for episode in range(args.episodes):
                episode_buffer = Experience_Buffer(args.buffer_size)
                state = self.env.state
                d = False
                reward_sum = 0 
                for step in range(args.max_episode_len):
                    if np.random.rand(1) < epsilon or total_step < args.pre_train_step:
                        action = np.random.randint(0,4)
                    else:
                        action = sess.run(
                            online_net['predict'],
                            feed_dict = {online_net['state']:[state]}
                            )[0]
                    state_1, reward, d = self.env.step(action)
                    episode_buffer.add(
                        np.reshape(np.array([state, action, reward, state_1, d]), [1,5])
                    )
                    total_step += 1 
                    if total_step > args.pre_train_step:
                        if epsilon > args.min_epsilon:
                            epsilon -= step_drop

                        batch_data = exp_buffer.sample(args.batch_size)
                        pre_action = sess.run(
                            online_net['predict'], 
                            feed_dict = {
                                online_net['state']:np.stack(batch_data[:, 3], axis=0)
                                }
                        )
                        Q_value_1 = sess.run(
                            target_net['Q_value'],
                            feed_dict = {
                                target_net['state']:np.stack(batch_data[:, 3], axis=0)
                                }
                        )
                        end_multiplier = -(batch_data[:,4] - 1)
                        double_Q = Q_value_1[range(args.batch_size), pre_action]
                        target_Q = batch_data[:,2] + args.gamma * double_Q * end_multiplier
                        _ = sess.run(
                            update_model,
                            feed_dict = {
                                online_net['state']:np.stack(batch_data[:, 0], axis=0),
                                target_Q_h:target_Q, actions_h:batch_data[:, 1]
                            }
                        )
                        if total_step % args.update_freq == 0:
                            self.update_target(target_Ops, sess)
                    reward_sum += reward
                    state = state_1
                    if d == True:
                        print(d)
                        continue
                exp_buffer.add(episode_buffer.buffer)
                r_list.append(reward_sum)

                mean_reward = np.mean(r_list[-10:])
                if len(r_list) % 10 == 0:
                    print('****', total_step, mean_reward, epsilon)

                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    model_path = os.path.join(args.save_path, 'agent.ckpt-%d-%.2f' % (episode, best_mean_reward))
                    saver.save(sess, model_path)
                    print("Saved Model in {}".format(model_path))

                    
def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--l_r', type=float, default=0.0001, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--save_path', type=str, default='./dqn', help='save model path')
    parser.add_argument('--env_config_path', type=str, default='env_config.yaml', help='save model path')
    parser.add_argument('--h_size', type=int, default=512, help='')
    parser.add_argument('--buffer_size', type=int, default=50000, help='')
    parser.add_argument('--pre_train_step', type=int, default=10000, help='')
    parser.add_argument('--update_freq', type=int, default=40, help='')
    parser.add_argument('--max_epsilon', type=float, default=1.0, help='')
    parser.add_argument('--min_epsilon', type=float, default=0.15, help='')
    parser.add_argument('--decay_step', type=float, default=10000, help='')
    parser.add_argument('--max_episode_len', type=int, default=80, help='')
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--load_model', type=bool, default=False)



    return parser.parse_args()
if __name__ == '__main__':

    args = Args()
    main(args).train()







    



