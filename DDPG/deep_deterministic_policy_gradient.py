import argparse
import os 
import random 
import numpy as np
from numpy.lib.function_base import gradient  
import tensorflow as tf
import tensorflow.contrib.slim as slim
import yaml
from utils import Experience_Buffer
from ENV.rebort_arm import game_env


class main():
    def __init__(self, args):
        self.args = args
        self.load_env()
        self.a_dim = self.env_config["action_dim"]
        self.s_dim = 3 + 2 * self.a_dim

        self.init_w = tf.contrib.layers.xavier_initializer()
        self.init_b = tf.constant_initializer(0.001)

    def load_env(self):
        
        with open(self.args.env_config_path, 'r', encoding='utf-8') as f:
            self.env_config = yaml.load(f)
        self.env = game_env(self.env_config)

    def actor_net(self, input, name):

        with tf.variable_scope(name):
            fc1 = slim.fully_connected(input, 200, activation_fn=tf.nn.relu6)
            fc2 = slim.fully_connected(fc1, 200, activation_fn=tf.nn.relu6)
            fc3 = slim.fully_connected(fc2, 10, activation_fn=tf.nn.relu)
            actions = slim.fully_connected(fc3, self.a_dim, activation_fn=tf.nn.tanh, scope='actions')
    
        return actions 

    def critic_net(self, input_s, input_a, name):

        with tf.variable_scope(name):
            fc_s = slim.fully_connected(input_s, 200, activation_fn=None)
            fc_a = slim.fully_connected(input_a, 200, activation_fn=None)
            sum_sa = tf.nn.relu6(tf.add(fc_s, fc_a))
            fc1 = slim.fully_connected(sum_sa, 200, activation_fn=tf.nn.relu6)
            fc2 = slim.fully_connected(fc1, 10, activation_fn=tf.nn.relu)
            q_value = slim.fully_connected(fc2, 1, activation_fn=None)
      
        return q_value


    def optimal_para(self, loss, var_list):

        optimizer = tf.train.RMSPropOptimizer(self.args.l_r)
        gradients = optimizer.compute_gradients(loss, var_list)

        return optimizer.apply_gradients(gradients)

    def update_target_graph(self, target_vars, online_vars):
        op_holder = []
        i = 0
        for onl_var, tar_var in zip(online_vars, target_vars):
            op_holder.append(
                target_vars[i].assign(
                    onl_var.value() * self.args.tau + tar_var.value() * (1-self.args.tau)
                )
            )
            i += 1
        return op_holder
    
    def update_target(self, op_holder, sess):

        for op in op_holder:
            sess.run(op)

    def train(self):

        args = self.args
        state_h = tf.placeholder(
            shape=[None, self.s_dim], dtype=tf.float32, name="state")
        state_h_ = tf.placeholder(
            shape=[None, self.s_dim], dtype=tf.float32, name="state_")
        target_q_h = tf.placeholder(shape=[None,1], dtype=tf.float32, name="taeget_q")

        action_h = tf.placeholder(shape=[None,self.a_dim], dtype=tf.float32, name="taeget_q")

        online_a = self.actor_net(state_h, "online_actor")
        target_a_ = self.actor_net(state_h_, "target_actor")

        online_q = self.critic_net(state_h, action_h, "online_critic")
        target_q_ = self.critic_net(state_h_, target_a_, "target_critic")

        o_actor_vars = tf.get_collection(tf.GraphKeys.VARIABLES, "online_actor")
        t_actor_vars = tf.get_collection(tf.GraphKeys.VARIABLES, "target_actor")
        o_critic_vars = tf.get_collection(tf.GraphKeys.VARIABLES, "online_critic")
        t_critic_vars = tf.get_collection(tf.GraphKeys.VARIABLES, "target_critic")

        loss_critic = tf.reduce_mean(tf.square(target_q_h - online_q))      
        optimal_on_cr = self.optimal_para(loss_critic, o_critic_vars)

        grads_qa = tf.gradients(online_q, action_h)
        policy_grads = tf.gradients(ys=online_a, xs=o_actor_vars, grad_ys=grads_qa)
        optimal_on_ac = tf.train.RMSPropOptimizer(-args.l_r).apply_gradients(zip(policy_grads, o_actor_vars))

        target_actor_ops = self.update_target_graph(t_actor_vars, o_actor_vars)
        target_critic_ops = self.update_target_graph(t_critic_vars, o_critic_vars)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
       
        exp_buffer = Experience_Buffer(args.buffer_size)
        with tf.Session(config=config) as sess:
            sess.run(init)
            if args.load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(args.save_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            total_step = 0
            var = args.var
            r_list = []
            best_mean_reward = 0
            for episode in range(args.episodes):
                state = self.env.reset()
                reward_sum = 0
                for step in range(args.max_episode_len):
                    if args.is_render:
                        self.env.render()
                    actions = sess.run(
                        online_a, 
                        feed_dict = {state_h:[state]}
                    )[0]
                    # add randomness to action selection for exploration
                    actions = np.clip(np.random.normal(actions, var), -1, 1) 
                    state_, reward, done = self.env.step(actions)

                    exp_buffer.add(
                        np.reshape(
                            np.array([state, actions, [reward], state_, done]), [1,5]
                        )
                    )
                    total_step += 1
                    if total_step > args.buffer_size:
                        # decay the action randomness
                        var = max([var * .9999, args.min_var])  
                        batch_data = exp_buffer.sample(args.batch_size)

                        q_value_ = sess.run(
                            target_q_, 
                            feed_dict = {state_h_: np.stack(batch_data[:, 3], axis=0)}
                        )
                        target_q_value = np.stack(batch_data[:, 2], axis=0) + args.gamma * q_value_
                        _, _ = sess.run(
                            [optimal_on_cr, optimal_on_ac],
                            feed_dict = {
                                target_q_h:target_q_value, 
                                state_h:np.stack(batch_data[:, 0], axis=0),
                                action_h:np.stack(batch_data[:, 1], axis=0)
                            }
                        )
                        if total_step % args.update_freq == 0:
                            self.update_target(target_actor_ops, sess)
                            self.update_target(target_critic_ops, sess)
                    state = state_
                    reward_sum += reward
         
                r_list.append(reward_sum)
                mean_reward = np.mean(r_list[-10:])
                result = "| done" if done else "| ----"
                print("Episode:", episode, 
                        result, 
                        "|Reward_sum:{}".format(reward_sum),
                        "Var:{}".format(var))
                 
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    model_path = os.path.join(
                        args.save_path, 'agent.ckpt-%d-%.2f' % (episode, best_mean_reward))
                    saver.save(sess, model_path)
                    print("Saved Model in {}".format(model_path))

def Args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000, help='')
    parser.add_argument('--buffer_size', type=int,
                        default=5000, help='experence buffer size')
    parser.add_argument('--env_config_path', type=str,
                        default='config/rebort_arm_config.yaml', help='the config of environment')
    parser.add_argument('--max_episode_len', type=int, default=200, help='')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of training online net')
    parser.add_argument('--l_r', type=float,
                        default=0.0001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='the discont rate of reward')
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--update_freq', type=int, default=10,
                        help='frequency of update target net')
    parser.add_argument('--var', type=float, default=2.0,
                        help='frequency of update target net')
    parser.add_argument('--min_var', type=float, default=0.05,
                        help='frequency of update target net')
    parser.add_argument('--is_render', type=bool, default=True)
    parser.add_argument('--save_path', type=str,
                        default='./DDPG/ckpt', help='save model path')
    parser.add_argument('--load_model', type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = Args()

    main(args).train()
    


        
