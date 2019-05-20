import tensorflow as tf

@tf.function
def demo(x,y):
	return x*x*y + y + 2


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

r = demo(x,y)