#Weights
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


#Bias
def init_bias(shape):
    init_bias = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias)

def conv1d(x,weights):
    #x is input accelration data and W is corresponding weight
    return tf.nn.conv1d(value=x,filters = weights,stride=1,padding='VALID')

def convolution_layer(input_x,shape):
   w1 = init_weights(shape)
   b = init_bias([shape[2]])
   return tf.nn.relu(conv1d(input_x,weights=w1)+b)


def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) +b

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32,shape=[None ,window_size,3]) #input tensor with 3 input channels
    y = tf.placeholder(tf.float32,shape=[None,6]) #Labels

with tf.variable_scope('net'):

    con_layer_1 = convolution_layer(x,shape=[4,3,32])#filter  of shape [filter_width, in_channels, out_channels]

    max_pool_1=tf.layers.max_pooling1d(inputs=con_layer_1,pool_size=2,strides=2,padding='Valid')

    con_layer_2 = convolution_layer(max_pool_1,shape=[4,32,64])

    max_pool_2 = tf.layers.max_pooling1d(inputs=con_layer_2,pool_size=2,strides=2,padding='Valid')

    flat = tf.reshape(max_pool_2,[-1,max_pool_2.get_shape()[1]*max_pool_2.get_shape()[2]])

    fully_conected = tf.nn.relu(normal_full_layer(flat,1024))


    second_hidden_layer = tf.nn.relu(normal_full_layer(fully_conected,512))
    hold_prob = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(second_hidden_layer,keep_prob=hold_prob)


    y_pred = normal_full_layer(full_one_dropout,6)
    pred_softmax = tf.nn.softmax(y_pred)
	
with tf.variable_scope('loss'):

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred))

with tf.variable_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)