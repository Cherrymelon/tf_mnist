import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# default parameters
learning_rate = 0.0005
sess = tf.InteractiveSession()
pooling_method ='max'
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'c:/tf_mnist/', 'Directory for storing data') # 第一次启动会下载文本资料

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# use xavier initializer
def weight_variable(name,shape):
    return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(name,shape):
    return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def pooling(x):
    if pooling_method == 'avg':
        return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    else:
       return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def batch_norm(input,is_training):
    input_shape = input.get_shape()
    axis = list(range(len(input_shape) - 1))
    shape = input_shape[-1:]
    gamma = tf.Variable(tf.ones(shape),name='gamma')
    beta = tf.Variable(tf.zeros(shape), name='beta')
    moving_mean = tf.Variable(tf.zeros(shape), name='moving_mean',
                              trainable=False)
    moving_variance = tf.Variable(tf.ones(shape),
                                  name='moving_variance',
                                  trainable=False)
    control_inputs = []
    def f1(): return 1
    def f0(): return 0
    flag = tf.cond(is_training,f1,f0)
    if flag==1:
        mean, variance = tf.nn.moments(input, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, self.decay)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, self.decay)
        control_inputs = [update_moving_mean, update_moving_variance]
    else:
        mean = moving_mean
        variance = moving_variance
    with tf.control_dependencies(control_inputs):
        return tf.nn.batch_normalization(
            input, mean=mean, variance=variance, offset=beta,
            scale=gamma, variance_epsilon=0.001)



def mnist_batch(batch_size):
    k = 60000 // batch_size
    for i in range(16000):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    _x: batch[0], _y: batch[1]})
                print("epoch"+str(k)+" :step: "+str(i)+ " training accuracy: "+str(train_accuracy))
            train_step.run(feed_dict={_x: batch[0], _y: batch[1], is_train : True})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        _x: mnist.test.images, _y: mnist.test.labels, is_train : False}))

_x = tf.placeholder('float', [None,784])
x = tf.reshape(_x, [-1,28,28,1])
_y = tf.placeholder('float', [None,10])
is_train = tf.placeholder('bool')

# conv_1 layer+pooling_1
w_conv1 = weight_variable('w_conv1', [3,3,1,64])
b_conv1 = bias_variable('b_conv1', [64])
feature_1 = conv2d(x, w_conv1)+b_conv1
z_1 = batch_norm(feature_1, is_train)
pooling_1 = pooling(tf.nn.relu(z_1))  # input: NWHC N*28*28*1 output: N*14*14*64

# conv2 layer+pooling_2
w_conv2 = weight_variable('w_conv2', [3,3,64,256])
b_conv2 = bias_variable('b_conv2', [256])
feature_2 = conv2d(pooling_1, w_conv2)+b_conv2
z_2 = batch_norm(feature_2, is_train)
pooling_2 = pooling(tf.nn.relu(z_2))   # input: NWHC N*14*14*64 output: N*7*7*256

# layer 3:full connect
w_fc1 = weight_variable('w_fc1', [7*7*256, 1024])
b_fc1 = bias_variable('b_fc1', [1024])
input_fc1 = tf.reshape(pooling_2, [-1, 7*7*256])
feature_3 = tf.matmul(input_fc1, w_fc1)+b_fc1
z_3 = batch_norm(feature_3, is_train)
output_fc1 = tf.nn.relu(z_3)  # output shape: W*1024

# layer 4:softmax layer   input:W*1024 output: 10-one-hat vector
w_fc2 = weight_variable('w_fc2', [1024, 10])
b_fc2 = bias_variable('b_fc2', [10])
ylogit = tf.matmul(output_fc1, w_fc2)+b_fc2
y = tf.nn.softmax(ylogit)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogit, labels=_y)


# train operation
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

mnist_batch(100)
