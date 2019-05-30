import tensorflow as tf

#Build computation graph (construction phase)
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

#Run the computation graph (execution phase)
#the execution phase generally run a loop that evaluates a training step repeatedly
#(for example one step per mini-batch) while gradually improving the model
init = tf.global_variables_initializer() #prepare an init node

with tf.Session() as sess:
    init.run()  #initialize all the variable at once
    result = f.eval() 

print(result)

sess.close()


'''
graph are automatically added to the default graph at you specifically avoid that
'''
#create a graph
graph = tf.Graph()

#make this graph the default graph (WARNING: this apply only in the with block)
with graph.as_default():
    x2 = tf.Variable(2)
    
print(x2.graph is graph) #the graph x2 belong to, is it graph?
print(x2.graph is tf.get_default_graph) #is it the default graph?

'''
Let see something interesting
'''
a = tf.constant(3)
b = a + 2
c = b + 5
d = b * 3

with tf.Session() as sess:
    c_eval = c.eval()
    d_eval = d.eval()
    print(c_eval)
    print(d_eval)
    
'''
In the previous code, b and a are evaluated twice when c and d are run/evaluated.
If you want to run/evaluate c and d efficiently whithout evaluating a and b twice, 
you must tel tensorflow to c and d in just one graph run.
'''
with tf.Session() as sess:
    c_eval, d_eval = sess.run([c, d])
    print(c_eval)
    print(d_eval)

'''
Note that in single-process tensorflow, multiple sessions do not share any state, even if they reuse
the same graph (each session would have its own copy of every variable).
In distributed tensorflow, variable state is stored on the servers, not in the sessions, 
so multiple sessions can share the same variable
'''


'''
placeholder:
placeholder is a way to hold a place for a variable that we will specify later on using
the feed_dict mechanism.
-we must specify the tensor data type
-optionally we can specify the shape (None means any size)
'''
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.Session() as sess:
    B1 = B.eval(feed_dict = {A: [[1,2,3]]})
    B2 = B.eval(feed_dict = {A: [[4,5,6], [7,8,9]]})
    
print(B1)
print(B2)
