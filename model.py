import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa


class MyEmbedding(layers.Layer):
    def __init__(self, dict):
        super(MyEmbedding, self).__init__()
        self.dict = dict
        self.embedding_char = layers.Embedding(input_dim=self.dict[0][0], output_dim=self.dict[0][1],
                                               name="char_embedding", trainable=True)
        self.embedding_bound = layers.Embedding(input_dim=self.dict[1][0], output_dim=self.dict[1][1],
                                                name="bound_embedding", trainable=True)
        self.embedding_flag = layers.Embedding(input_dim=self.dict[2][0], output_dim=self.dict[2][1],
                                               name="flag_embedding", trainable=True)
        self.embedding_radical = layers.Embedding(input_dim=self.dict[3][0], output_dim=self.dict[3][1],
                                                  name="radical_embedding", trainable=True)
        self.embedding_pinyin = layers.Embedding(input_dim=self.dict[4][0], output_dim=self.dict[4][1],
                                                 name="pinyin_embedding", trainable=True)

    def call(self, input, traininf=None):
        embedding = []
        embedding.append(self.embedding_char(input[0]))
        embedding.append(self.embedding_bound(input[1]))
        embedding.append(self.embedding_flag(input[2]))
        embedding.append(self.embedding_radical(input[3]))
        embedding.append(self.embedding_pinyin(input[4]))
        embed = tf.concat(embedding, axis=-1)
        # print(embed.shape)
        return embed


class MyLSTM(layers.Layer):  ###定义为layer，需要定义为Model，才能使用summary
    def __init__(self, hidden_dim):
        super().__init__(self)

        self.fw_lstm1 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=False,
                                    dropout=0.4, name="fwd_lstm1", trainable=True)
        self.bw_lstm1 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=True,
                                    dropout=0.4, name="bwd_lstm1", trainable=True)
        self.fw_lstm2 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=False,
                                    dropout=0.4, name="fwd_lstm2", trainable=True)
        self.bw_lstm2 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=True,
                                    dropout=0.4, name="bwd_lstm2", trainable=True)

    def call(self, inputs, training=None):
        fw_lstm1 = self.fw_lstm1(inputs)
        bw_lstm1 = self.bw_lstm1(inputs)
        output = tf.concat([fw_lstm1, bw_lstm1], axis=-1)
        # print(output.shape)
        fw_lstm2 = self.fw_lstm2(output)
        bw_lstm2 = self.bw_lstm2(output)
        outputs = tf.concat([fw_lstm2, bw_lstm2], axis=-1)
        return outputs


def Mutiple_H(x):
    dim = 4
    assert tf.shape(x)[-1] % dim == 0
    output = tf.expand_dims(x, axis=-2)
    y = tf.concat(tf.split(output, num_or_size_splits=dim, axis=-1), axis=-2)
    y = tf.transpose(y, [0, 2, 1, 3])
    return y


# 自注意力机制层
class Self_Attention_Layer(layers.Layer):
    def __init__(self, attention_size):
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
        self.dense = layers.Dense(self.attention_size, trainable=True)

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
            QK = tf.matmul(Q, tf.transpose(K, [0, 1, 3,
                                               2]))  # 现在QK的大小是[batch_size, dim, max_sen_len,max_sen_len]转置位置K各个维度，好做点乘
            if is_mask:
                # 接下来实现带有mask操作的自注意力机制,之前尝试使用句子的长度来做mask没有弄成，现在再次尝试
                mask = tf.sequence_mask(sen_len,
                                        maxlen=max_sen_len)  # sen_len.shape+(max_sen_len) = (sen_len,max_sen_len)
                mask = tf.expand_dims(tf.expand_dims(mask, 1),
                                      axis=1)  # mask主要是将填充的地方的权值设置的非常小，这样在加权的时候就会是填充的单词起到作用了(sen_len,1,1,max_sen_len)
                # tf.tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
                mask = tf.tile(mask, [1, tf.shape(QK)[1], tf.shape(QK)[2],
                                      1])  # 现在有了mask矩阵，下面开始将pading的单词的权重是设置的很小(sen_len, dim, max_sen_len,max_sen_len)
                padding_val = -2 ** 32  # 使padding为0的数变得很小，在做softmax时，以至于填充出的归一化概率值很大
                # 不让分值随着维度过大，点乘和太大，为了把注意力矩阵变成标准正态分布，让softmat归一化之后的结果更加稳定，以便反向传播获取更加平衡的梯度
                #以数组为例，2个长度是len，均值是0，方差是1的数组点积会生成长度是len，均值是0，方差是len的数组  D（sigmaX）=sigma D（X），sigma是求和。
                # 而方差变大会导致softmax的输入推向正无穷或负无穷，这时的梯度会无限趋近于0，不利于训练的收敛。因此除以len的开方，
                # 可以是数组的方差重新回归到1，有利于训练的收敛。
                QK = tf.where(mask, QK, tf.ones_like(QK) * padding_val) / tf.sqrt(  #将bool为False的用后面对应位置数据替换
                    tf.cast(self.attention_size, dtype=tf.float32))  # 采用的是缩放的点积
                QK = self.softmax(QK)
                Z = tf.matmul(QK, V)
                Z = tf.reshape(Z, [tf.shape(inputs)[0], max_sen_len, -1])
                Z = self.dropout(Z)
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
                mask = tf.sequence_mask(sen_len,
                                        maxlen=max_sen_len)  # sen_len.shape+(max_sen_len) = (sen_len,max_sen_len)
                mask = tf.expand_dims(mask, 1)  # mask主要是将填充的地方的权值设置的非常小，这样在加权的时候就会是填充的单词起到作用了(sen_len,1,max_sen_len)
                # tf.tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
                mask = tf.tile(mask, [1, tf.shape(QK)[1],
                                      1])  # 现在有了mask矩阵，下面开始将pading的单词的权重是设置的很小(sen_len,max_sen_len,max_sen_len)
                padding_val = -2 ** 32  # 使padding为0的数变得很小，在做softmax时，以至于填充出的归一化概率值很大
                # 不让分值随着维度过大，点乘和太大，为了把注意力矩阵变成标准正态分布，让softmat归一化之后的结果更加稳定，以便反向传播获取更加平衡的梯度
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


class Crf_layer(layers.Layer):
    def __init__(self, label_nums):
        super().__init__(self)
        self.label_num = label_nums
        self.dense = layers.Dense(self.label_num, use_bias=False, trainable=True,
                                  kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.tran_paras = tf.Variable(tf.random.uniform(shape=(self.label_num, self.label_num)), trainable=True)

    def call(self, inputs, targets, lens, tran_paras, training=None):
        '''inputs是经过（Self_Attention_layer）或LSTM层之后的输出，这里还要将输入的大小[batch_size,maxlen,attention_dim]调整为
        [batch_size,maxlen,label_nums]'''
        out = self.dense(inputs)  # 调整大小为[batch_size,maxlen,nums_label]

        if training:
            self.log_likelihood, self.tran_paras = \
                tfa.text.crf_log_likelihood(out, targets, lens, transition_params=self.tran_paras)
            self.batch_pred_sequence, self.batch_viterbi_score = tfa.text.crf_decode(out, self.tran_paras, lens)

            self.loss = tf.reduce_mean(-self.log_likelihood)

            return self.loss, self.batch_pred_sequence, self.tran_paras
        else:
            self.log_likelihood, tran_paras = \
                tfa.text.crf_log_likelihood(out, targets, lens, tran_paras)
            self.batch_pred_sequence, _ = tfa.text.crf_decode(out, tran_paras, lens)
            self.loss = tf.reduce_mean(-self.log_likelihood)

            return self.loss, self.batch_pred_sequence


class Model_NER(tf.keras.Model):
    def __init__(self, dict, hide_dim, label_num, attention_size):
        super().__init__(self)
        self.embeddinglayer = MyEmbedding(dict)
        self.rnn_layer = MyLSTM(hide_dim)
        self.crf_layer = Crf_layer(label_num)
        self.self_attention = Self_Attention_Layer(attention_size)

    def call(self, traintext, label, tran_paras, training=None):
        # 计算一下长度
        traintext = tf.convert_to_tensor(traintext)
        label = tf.convert_to_tensor(label)
        self.lens = tf.reduce_sum(tf.sign(traintext[0]), axis=-1)
        self.out = self.embeddinglayer(traintext)
        self.out = self.rnn_layer(self.out)
        self.out = self.self_attention(self.out, self.lens)
        if training:
            self.loss, self.batch_pred_seq, self.tran_paras = self.crf_layer(self.out, label, self.lens, tran_paras)

            return self.loss, self.batch_pred_seq, self.lens, self.tran_paras
        else:
            self.loss, self.batch_pred_seq = self.crf_layer(self.out, label, self.lens, tran_paras)

            return self.loss, self.batch_pred_seq, self.lens


if __name__ == "__main__":
    model = Self_Attention_Layer(400)
    x = tf.Variable(tf.random.uniform([5, 200, 800], -1, 1))
    len = tf.constant([200, 198, 189, 196, 187], dtype=tf.int32)
    Z = model(x, len, Mutiple_heard=False)
    print(Z)
