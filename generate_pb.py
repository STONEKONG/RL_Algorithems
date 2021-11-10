import tensorflow as tf 
from tensorflow.python.framework import graph_util


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''

    output_node_names = "target_net/Q_value"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() 

    for op in graph.get_operations():
        print(op.name)

    input_graph_def = graph.as_graph_def() 
    print(input_graph_def.node)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) 
        output_graph_def = graph_util.convert_variables_to_constants( 
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f: 
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node)) 

if __name__ == "__main__":  

    
    ckpt_path = 'dqn/agent.ckpt-4015-26.60'
    pb_path = 'agent.ckpt-4015-26.60.pb'

    freeze_graph(ckpt_path, pb_path)
