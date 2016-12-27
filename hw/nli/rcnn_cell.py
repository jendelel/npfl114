import tensorflow as tf
import numpy as np


class RCNNCell(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self, ctx_dim, emb_dim, func, scope = None):
        self.scope = scope
        self.ctx_dim = ctx_dim
        self.emb_dim = emb_dim
        self.func = func
        self.first = True
        with tf.variable_scope(scope or "rcnn_cell"):
            self.w = tf.get_variable("W", [ctx_dim, ctx_dim], tf.float32)
            self.initial_state = tf.get_variable("inital_state", [ctx_dim], tf.float32)
            self.w_s = tf.get_variable("Ws", [emb_dim, ctx_dim], tf.float32)


    @property
    def state_size(self):
        return self.ctx_dim

    @property
    def output_size(self):
        return self.ctx_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "rcnn_cell"):
            last_context = state
            print("last_context", last_context.get_shape())
            print("W", self.w.get_shape())
            print("Ws", self.w_s.get_shape())
            r1 = tf.matmul(last_context, self.w)
            print("r1", r1.get_shape())
            r2 = tf.matmul(inputs, self.w_s)
            print("r2", r2.get_shape())
            next_outputs = self.func(r1 + r2)
            print("next_outputs", next_outputs.get_shape())
            print("inputs", inputs.get_shape())
            return (last_context, next_outputs)

    def zero_state(self, batch_size, dtype):
        t = tf.tile(self.initial_state, [batch_size])
        return tf.reshape(t, [batch_size, self.ctx_dim])


