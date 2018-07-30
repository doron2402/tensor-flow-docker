import tensorflow as tf

x = tf.get_variable('x', shape=[5,2,3])
y = tf.get_variable('y', shape=[5,3,1])

z = tf.matmul(x,y, name='z')
assert z.shape == (5,2,1)

tf.add_to_collection('basic_collection', z)
assert z is tf.get_collection('basic_collection')[0]

default_graph = tf.get_default_graph()
assert x.graph is default_graph
assert y.graph is default_graph
assert z.graph is default_graph

print('test passed')

