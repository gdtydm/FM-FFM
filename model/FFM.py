import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from util import dataGenerate


def Model(xtrain, ytrain,field_dict, steps = 1000,learning_rate = 0.01,K = 3, display_information = 100,ffm = True, seed = 0):
    tf.set_random_seed(seed)
    n = xtrain.shape[1]
    f = sorted(field_dict.items(), key=lambda s:s[1], reverse=True)[0][1]
    X = tf.placeholder(tf.float32, shape=[None, n], name="X") # (None, feature_size)
    y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
    V = tf.get_variable("v", shape=[f+1, n, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.3)) #(fields, feature_size, K)
    W = tf.get_variable("Weights", shape=[n, 1], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.3))
    b = tf.get_variable("Biases", shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    logits = tf.matmul(X, W) + b
    if ffm:
        # FFM 部分
        fm_hat = tf.constant(0, dtype='float32')
        for i in range(n):
            for j in range(i+1, n):
                fm_hat += tf.multiply(tf.reduce_sum(tf.multiply(V[field_dict[j],i], V[field_dict[i],j])), tf.reshape(tf.multiply(X[:,i], X[:,j]), [-1,1]))

        logits = logits + fm_hat
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

    y_hat = tf.nn.sigmoid(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            loss_,_,y_h = sess.run([loss,optimizer, y_hat], feed_dict={X:xtrain, y:ytrain})
            if i % display_information == 0:
                print("Train accuracy is %.6f loss is %.6f" % (accuracy_score(ytrain.reshape(-1,), np.where(np.array(y_h).reshape(-1,) >= 0.5,1,0)),
                                                               loss_))

if __name__ == "__main__":
    x_train, y_train, field_dict = dataGenerate()
    Model(x_train, y_train, field_dict,1000, 0.01, ffm=True)
