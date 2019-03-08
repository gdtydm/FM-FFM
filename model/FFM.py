import tensorflow as tf
import numpy as np
class FFM(object):
    def __init__(self, hparams, df_i, df_v):
        # df_i, df_v  None * n
        self.hparams = hparams
        tf.set_random_seed(self.hparams.seed)
        self.line_result = self.line_section(df_i, df_v)
        self.fm_result = self.fm_section(df_i, df_v)
        print(self.line_result, self.fm_result)
        self.logits = self.line_result + self.fm_result

    def line_section(self, df_i, df_v):
        with tf.variable_scope("line"):
            weights = tf.get_variable("weights",
                                      shape=[self.hparams.feature_nums, 1],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.glorot_uniform()) # f * 1
            batch_weights = tf.nn.embedding_lookup(weights, df_i) # none * n * 1
            batch_weights = tf.squeeze(batch_weights, axis=2) # None * n
            line_result = tf.multiply(df_v, batch_weights, name="line_w_x") # none * n
            biase =  tf.get_variable("biase",
                                    shape=[1, 1],
                                    dtype=tf.float32,
                                    initializer=tf.initializers.zeros()) # 1 * 1
            line_result = tf.add(tf.reduce_sum(line_result, axis=1, keepdims=True), biase) #  None，1
        return line_result


    def fm_section(self, df_i, df_v):
        with tf.variable_scope("fm"):
            embedding = tf.get_variable("embedding",
                                        shape=[self.hparams.field_nums,
                                               self.hparams.feature_nums,
                                               self.hparams.embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.random_normal()) # field * f * embedding_size
            fm_result = None
            for i in range(self.hparams.field_nums):
                for j in range(i+1, self.hparams.field_nums):
                    vi_fj = tf.nn.embedding_lookup(embedding[j], df_i[:,i]) #  None * embedding_size
                    vj_fi = tf.nn.embedding_lookup(embedding[i], df_i[:,j]) #  None * embedding_size
                    wij = tf.multiply(vi_fj, vj_fi)

                    x_i = tf.expand_dims(df_v[:,i], 1) # None * 1
                    x_j = tf.expand_dims(df_v[:,j], 1) # None * 1
                    
                    xij = tf.multiply(x_i, x_j)  # None * 1
                    if fm_result is None:
                        fm_result = tf.reduce_sum(tf.multiply(wij, xij), axis=1, keepdims=True)
                    else:
                        fm_result += tf.reduce_sum(tf.multiply(wij, xij), axis=1, keepdims=True)

            fm_result = tf.reduce_sum(fm_result, axis=1, keep_dims=True)
        return fm_result








