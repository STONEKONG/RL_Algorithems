import tensorflow as tf 
from tensorflow.python.framework import graph_util
import argparse

def freeze_graph(args):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''

   
    saver = tf.train.import_meta_graph(args.ckpt_path + '.meta', clear_devices=True)
    graph = tf.get_default_graph() 
    
    input_graph_def = graph.as_graph_def() 
    with tf.Session() as sess:
        saver.restore(sess, args.ckpt_path) 
        output_graph_def = graph_util.convert_variables_to_constants( 
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=args.output_node_name.split(","))
 
        with tf.gfile.GFile(args.save_path, "wb") as f: 
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node)) 

def Args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='./DDPG/ckpt/agent.ckpt-3243-55.03', help='ckpt model path')
    parser.add_argument('--save_path', type=str,
                        default='./DDPG/ckpt/agent.ckpt-3243-55.03.pb', help='save model path')
    parser.add_argument('--output_node_name', type=str,
                        default="target_actor/fully_connected_3/Tanh", help='save model path')
    return parser.parse_args()

if __name__ == "__main__":  

    
    args = Args()
    freeze_graph(args)
