import tensorflow as tf

try: 
    x = tf.get_variable('x', shape=[5,2,3])
except ValueError:
    print('Error assiging value to `x`')

g = tf.Graph()
default_graph = tf.get_default_graph()
with g.as_default():
    x = tf.get_variable('x', shape=[5,2,3])
    print('Assignment in new graph context runs without error')
    assert x.graph is g
    assert x.graph is not default_graph
    print('All tests passed!')




