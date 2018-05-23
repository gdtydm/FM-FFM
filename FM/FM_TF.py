import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from util import dataGenerate

# xtrain = load_iris()['data']
# ytrain = load_iris()['target']
#
#
# xtrain = np.array(xtrain[ytrain != 2])
# ytrain = np.array(ytrain[ytrain != 2]).reshape(-1,1)
#
# DATA = "train_data.txt"
# def loadDataSet(data):
#     '''导入训练数据
#     input:  data(string)训练数据
#     output: dataMat(list)特征
#             labelMat(list)标签
#     '''
#     dataMat = []
#     labelMat = []
#     fr = open(data)  # 打开文件
#     for line in fr.readlines():
#         lines = line.strip().split("\t")
#         lineArr = []
#
#         for i in range(len(lines) - 1):
#             lineArr.append(float(lines[i]))
#         dataMat.append(lineArr)
#         labelMat.append(float(lines[-1]))  # 转换成{-1,1}
#
#
#     fr.close()
#     return np.array(dataMat), np.array(labelMat).reshape(-1,1)
#



def Model(xtrain, ytrain, steps = 1000,learning_rate = 0.01,K = 3, display_information = 100,fm = True, seed = 0):
    tf.set_random_seed(seed)
    n = xtrain.shape[1]

    X = tf.placeholder(tf.float32, shape=[None, n], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
    V = tf.get_variable("v", shape=[n, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.3))
    W = tf.get_variable("Weights", shape=[n, 1], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.3))
    b = tf.get_variable("Biases", shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    logits = tf.matmul(X, W) + b
    if fm:
        # FM 部分
        fm_hat = tf.reduce_sum(np.square(tf.matmul(X , V)) - (tf.matmul(tf.multiply(X,X), tf.multiply(V,V))), axis=1, keep_dims=True) / 2
        # fm_hat = tf.constant(0, dtype='float32')
        # for i in range(n):
        #     for j in range(i+1, n):
        #         fm_hat += tf.multiply(tf.reduce_sum(tf.multiply(V[i], V[j])), tf.reshape(tf.multiply(X[:,i], X[:,j]), [-1,1]))

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
    # dataTrain, labelTrain = loadDataSet("train_data.txt")
    x_train,y_train,field_dict = dataGenerate()


    Model(x_train, y_train, 1000, 0.01, fm=True)




