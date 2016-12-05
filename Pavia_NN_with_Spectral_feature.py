import tensorflow as tf
import numpy
#import tensorflow.examples.tutorials.mnist.input_data as input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Pavia_X = numpy.loadtxt(open("train_x.csv","rb"),delimiter=",",skiprows=0);
Pavia_Y = numpy.loadtxt(open("train_y.csv","rb"),delimiter=",",skiprows=0);
train_x=Pavia_X[0:10000,:]
train_y=Pavia_Y[0:10000,:]
test_x=Pavia_X[0:40000,:]
test_y=Pavia_Y[0:40000,:]
if __name__ == "__main__":
    print ("Pavia_X.shape",Pavia_X.shape)
    print ("Pavia_Y.shape",Pavia_Y.shape)
    print ("train_x.shape",train_x.shape)
    print ("train_y.shape",train_y.shape)
    print ("test_x.shape",test_x.shape)
    print ("test_y.shape",test_y.shape)


x = tf.placeholder(tf.float32, [None, 103])
y_actual = tf.placeholder(tf.float32, shape=[None, 9])
W = tf.Variable(tf.zeros([103,9]))        #初始化权值W
b = tf.Variable(tf.zeros([9]))            #初始化偏置项b
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)     #加权变换并进行softmax回归，得到预测概率
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))   #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):               #训练阶段，迭代1000次
        #batch_xs, batch_ys = tf.train.shuffle_batch([train_x,train_y],100,50000,10000)
        print (i)
        sess.run(train_step, feed_dict={x: train_x, y_actual: train_y})   #执行训练
        #if(i%100==0):                  #每训练100次，测试一次
        print ("accuracy:",sess.run(accuracy, feed_dict={x: train_x, y_actual: train_y}))
            



