import tensorflow as tf


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
print(f.eval)


w = tf.constant(3)
x = w + 2
y = x + 5
z = x*3

print(y.eval)
print(z.eval)