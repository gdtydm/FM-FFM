import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

class FM(object):
    def __init__(self, hparams, df_i, df_v):
        # df_i, df_v  None * n
        self.hparams = hparams
        self.load_activation_fn()

        self.line_result = self.line_section(df_i, df_v)
        self.fm_result = self.fm_section(df_i, df_v)
        if not self.hparams.use_deep:
            assert self.line_result.shape == self.fm_result.shape
            self.logits = self.line_result + self.fm_result
        else:
            self.deep_result = self.deep_section()
            self.deep_result = tf.nn.dropout(self.deep_result, keep_prob=self.hparams.deep_output_keep_dropout)

            self.line_result = self.deep_section()
            self.line_result = tf.nn.dropout(self.line_result, keep_prob=self.hparams.line_output_keep_dropout)

            self.fm_result = self.deep_section()
            self.fm_result = tf.nn.dropout(self.fm_result, keep_prob=self.hparams.fm_output_keep_dropout)

            concat = tf.concat(values=[self.line_result, self.fm_result, self.deep_result], axis=1)
            
            self.logits = tf.layers.dense(concat, units=1, activation=None)

    def line_section(self, df_i, df_v):
        with tf.variable_scope("line"):
            weights = tf.get_variable("weights",
                                      shape=[self.hparams.field_nums, 1],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.glorot_uniform()) # f * 1
            batch_weights = tf.nn.embedding_lookup(weights, df_i) # none * n * 1
            batch_weights = tf.squeeze(batch_weights, axis=2)
            line_result = tf.multiply(df_v, batch_weights, name="line_w_x") # none * n
            if self.hparams.use_deep:
                return line_result
            biase =  tf.get_variable("biase",
                                    shape=[1, 1],
                                    dtype=tf.float32,
                                    initializer=tf.initializers.zeros()) # f * 1
            line_result = tf.add(tf.reduce_sum(line_result, axis=1, keepdims=True), biase)
        return line_result        


    def fm_section(self, df_i, df_v):
        with tf.variable_scope("fm"):
            embedding = tf.get_variable("embedding",
                                        shape=[self.hparams.field_nums,
                                               self.hparams.embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.random_normal()) # f * embedding_size
            batch_embedding = tf.nn.embedding_lookup(embedding, df_i) # none * n * embedding_size
            df_v = tf.expand_dims(df_v, axis=2) # none * n * 1
            self.xv = tf.multiply(df_v, batch_embedding) # none * n * embedding_size
            sum_square = tf.square(tf.reduce_sum(self.xv, axis=1)) # none * embedding_size

            square_sum = tf.reduce_sum(tf.square(self.xv), axis=1) # none * embedding_size
            
            fm_result = 0.5 * tf.subtract(sum_square, square_sum)
            if self.hparams.use_deep:
                return fm_result
            
            fm_result = tf.reduce_sum(fm_result, axis=1, keep_dims=True)
        return fm_result


    def deep_section(self):
        deep_input = tf.reshape(self.xv, [-1, self.hparams.columns * self.hparams.embedding_size], name="deep_input")
        deep_input = tf.nn.dropout(x=deep_input, keep_prob=self.hparams.deep_input_keep_dropout)
        for i,v in enumerate(self.hparams.layers):
            deep_input = tf.layers.dense(deep_input, units=v, activation=None)
            if self.hparams.use_batch_normal:
                deep_input = tf.layers.batch_normalization(deep_input)
            deep_input = self.activation(deep_input)
            if (i+1) != len(self.hparams.layers):
                deep_input = tf.nn.dropout(deep_input, self.hparams.deep_mid_keep_dropout)
        return deep_input
    

    def load_activation_fn(self):
        if self.hparams.activation == "relu":
            self.activation = tf.nn.relu
        elif self.hparams.activation == "tanh":
            self.activation = tf.nn.tanh
        elif self.hparams.activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        elif self.hparams.accuracy == "elu":
            self.activation = tf.nn.elu
        else:
            raise ValueError("please input correct activat func.")







