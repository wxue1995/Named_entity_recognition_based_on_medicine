import random
import tensorflow as tf

#
# def Z(a):
#     return a * (4 ** 2 + 4)
#
#
# def Y(a, x):
#     return a * (x ** 2 + 4)
#
#
# x = tf.Variable(initial_value=tf.random.truncated_normal(shape=(2,2)), dtype=tf.float32)
# print(x)
# optimizer = tf.keras.optimizers.Adam(0.01)
#
# for _ in range(5000):
#     input = random.randint(1, 255)
#     target = Z(input)
#     with tf.GradientTape() as tape:
#         pre = Y(input, x)
#         loss = tf.reduce_sum(abs(pre - target))
#
#     gradient = tape.gradient(loss, x)
#     optimizer.apply_gradients([(gradient, x)])
#
#     if _ % 100 == 0:
#         print("loss:", loss.numpy())
# print("x: ", x.numpy())
from tensorflow.keras import layers


def Mutiple_H(x):
    dim = 4
    assert tf.shape(x)[-1] % dim == 0
    output = tf.expand_dims(x, axis=-2)
    y = tf.concat(tf.split(output, num_or_size_splits=dim, axis=-1), axis=-2)
    y = tf.transpose(y, [0, 2, 1, 3])
    return y


# 自注意力机制层
class Self_Attention_Layer(layers.Layer):
    def __init__(self,attention_size):
        super().__init__(self)
        self.attention_size = attention_size
        self.dense_Q = layers.Dense(self.attention_size, use_bias=False, trainable=True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dense_K = layers.Dense(self.attention_size, use_bias=False, trainable=True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dense_V = layers.Dense(self.attention_size, use_bias=False, trainable=True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dropout = layers.Dropout(0.4, name='attenion_drop')
        self.softmax = layers.Softmax()
        self.dense = layers.Dense(self.attention_size)
    def call(self, inputs, sen_len, is_mask=True, Mutiple_heard=True):
        # 就算QKV
        Q = self.dense_Q(inputs)
        K = self.dense_K(inputs)
        V = self.dense_V(inputs)
        max_sen_len = tf.shape(inputs)[1]
        if Mutiple_heard:
            Q = Mutiple_H(Q)
            K = Mutiple_H(K)
            V = Mutiple_H(V)
            # 下面开始做注意力机制,如果使用mask操作，还要用到句子的长度,不使用mask操作会简单很多
            QK = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))  # 现在QK的大小是[batch_size, dim, max_sen_len,max_sen_len]转置位置K各个维度，好做点乘
            if is_mask:
                # 接下来实现带有mask操作的自注意力机制,之前尝试使用句子的长度来做mask没有弄成，现在再次尝试
                mask = tf.sequence_mask(sen_len, maxlen=max_sen_len)  #sen_len.shape+(max_sen_len) = (sen_len,max_sen_len)
                mask = tf.expand_dims(tf.expand_dims(mask, 1),axis=1)  # mask主要是将填充的地方的权值设置的非常小，这样在加权的时候就会是填充的单词起到作用了(sen_len,1,1,max_sen_len)
                # tf.tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
                mask = tf.tile(mask, [1,tf.shape(QK)[1], tf.shape(QK)[2], 1])  # 现在有了mask矩阵，下面开始将pading的单词的权重是设置的很小(sen_len, dim, max_sen_len,max_sen_len)
                padding_val = -2 ** 32  # 使padding为0的数变得很小，在做softmax时，以至于填充出的归一化概率值很大
                # 不让分值随着维度过大，点乘和太大，为了把注意力矩阵变成标准正态分布，让softmat归一化之后的结果更加稳定，以便反向传播获取更加平衡的梯度
                QK = tf.where(mask, QK, tf.ones_like(QK) * padding_val) / tf.sqrt(  ##将bool为Flast的用后面对应位置数据替换
                    tf.cast(self.attention_size, dtype=tf.float32))  # 采用的是缩放的点积
                QK = self.softmax(QK)
                Z = tf.matmul(QK, V)
                Z = tf.reshape(Z, [tf.shape(inputs)[0],max_sen_len,-1])
                Z =self.dropout(Z)
                Z = self.dense(Z)

            else:
                # 不使用mask操作，还要有个缩放因子
                QK = self.softmax(QK / tf.sqrt(self.attention_size))
                # softmax之后就是加权求和输出z，很简单的矩阵乘法
                Z = tf.matmul(QK, V)  # 使用这个矩阵乘法之后，默认在最后两个维度进行做乘法，也就是加权求和了
                Z = tf.reshape(Z, [tf.shape(inputs)[0], max_sen_len, -1])
                Z = self.dropout(Z)
                Z = self.dense(Z)
            return Z
        else:
            # 下面开始做注意力机制,如果使用mask操作，还要用到句子的长度,不使用mask操作会简单很多
            QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # 现在QK的大小是[batch_size,max_sen_len,max_sen_len]转置位置K各个维度，好做点乘
            if is_mask:
                # 接下来实现带有mask操作的自注意力机制,之前尝试使用句子的长度来做mask没有弄成，现在再次尝试
                mask = tf.sequence_mask(sen_len, maxlen=max_sen_len)  #sen_len.shape+(max_sen_len) = (sen_len,max_sen_len)
                mask = tf.expand_dims(mask, 1)  # mask主要是将填充的地方的权值设置的非常小，这样在加权的时候就会是填充的单词起到作用了(sen_len,1,max_sen_len)
                # tf.tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
                mask = tf.tile(mask, [1, tf.shape(QK)[1], 1])  # 现在有了mask矩阵，下面开始将pading的单词的权重是设置的很小(sen_len,max_sen_len,max_sen_len)
                padding_val = -2 ** 32  # 使padding为0的数变得很小，在做softmax时，以至于填充出的归一化概率值很大
                #不让分值随着维度过大，点乘和太大，为了把注意力矩阵变成标准正态分布，让softmat归一化之后的结果更加稳定，以便反向传播获取更加平衡的梯度
                QK = tf.where(mask, QK, tf.ones_like(QK) * padding_val) / tf.sqrt(  ##将bool为Flast的用后面对应位置数据替换
                    tf.cast(self.attention_size, dtype=tf.float32))  # 采用的是缩放的点积,
                QK = self.softmax(QK)
                Z = tf.matmul(QK, V)
                Z = self.dropout(Z)
                Z = self.dense(Z)
            else:
                # 不使用mask操作，还要有个缩放因子
                QK = self.softmax(QK / tf.sqrt(self.attention_size))
                # softmax之后就是加权求和输出z，很简单的矩阵乘法
                Z = tf.matmul(QK, V)  # 使用这个矩阵乘法之后，默认在最后两个维度进行做乘法，也就是加权求和了
                Z = self.dropout(Z)
                Z = self.dense(Z)
            return Z


model = Self_Attention_Layer(400)
x = tf.Variable(tf.random.uniform([5, 200, 800], -1, 1))
len = tf.constant([200,198,189,196,187], dtype=tf.int32)
Z = model(x,len,Mutiple_heard=False)
print(Z)

# tran_paras = tf.Variable(tf.fill(dims=(31, 31),value=0.5))
# print(tran_paras)



# shap = [5, 30]

# y = Mutiple_H(x)
# print(y.shape)


# min = tf.constant([[[[1,2,1],[1,3,3],[1,5,3],[1,3,4]],[[1,2,1],[1,1,1],[1,5,1],[1,4,1]]]])
# m = tf.reshape(min,[1,4,-1])
# print(min,min.shape)
#
# print(m)
# print(tf.reshape(min,[1,-1,3]))