from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Input, Embedding, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.initializers import glorot_normal, Zeros, TruncatedNormal
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf

class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        self.embedding = Embedding(input_dim=self.total_dim, output_dim=self.embed_dim)

    def call(self, x):
        x = x + tf.constant(self.offsets, dtype=tf.int32)
        return self.embedding(x)

class MultiHeadSelfAttention(Layer):
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling

    def build(self, input_shape):
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(shape=(embedding_size, self.att_embedding_size * self.head_num), initializer=TruncatedNormal())
        self.W_key = self.add_weight(shape=(embedding_size, self.att_embedding_size * self.head_num), initializer=TruncatedNormal())
        self.W_Value = self.add_weight(shape=(embedding_size, self.att_embedding_size * self.head_num), initializer=TruncatedNormal())
        if self.use_res:
            self.W_Res = self.add_weight(shape=(embedding_size, self.att_embedding_size * self.head_num), initializer=TruncatedNormal())

    def call(self, inputs):
        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))
        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        normalized_att_scores = tf.nn.softmax(inner_product)
        result = tf.matmul(normalized_att_scores, values)
        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        return tf.nn.relu(result)

class AutoIntPlus(Layer):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_units=[128, 64, 32], **kwargs):
        super(AutoIntPlus, self).__init__(**kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.int_layers = [MultiHeadSelfAttention(embedding_size, att_head_num, att_res) for _ in range(att_layer_num)]
        self.dnn = [Dense(units, activation='relu') for units in dnn_units]
        self.final_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        att_input = self.embedding(inputs)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = Flatten()(att_input)
        for layer in self.dnn:
            att_output = layer(att_output)
        return self.final_layer(att_output)

class AutoIntPlusModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_units=[128, 64, 32], **kwargs):
        super(AutoIntPlusModel, self).__init__(**kwargs)
        self.autoIntPlus_layer = AutoIntPlus(field_dims, embedding_size, att_layer_num, att_head_num, att_res, dnn_units)

    def call(self, inputs):
        return self.autoIntPlus_layer(inputs)

#
