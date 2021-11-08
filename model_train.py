import os
from tqdm import tqdm
from metrics import calculatePRF1
from matplotlib import pyplot as plt
from data_utils import BatchManager, get_dict
from model import Model_NER
import tensorflow as tf
import time
import matplotlib
# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
dict_file = 'data/prepare/dict.pkl'
mapping_dict = get_dict(dict_file)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_and_test(batch_size, dict, hide_dim, label_num, attention_dim):
    model = Model_NER(dict, hide_dim, label_num, attention_dim)

    savepath = "checkpoints/"
    if not os.path.exists(savepath):
        print('-----make save_model savepath----')
        os.makedirs(savepath)
    else:

        model.load_weights(os.path.join(savepath, 'ckpt'))
        print('-----load save_model weight----')

    train_manager = BatchManager(batch_size=batch_size, name='train')
    train_leght = train_manager.len_data
    test_manager = BatchManager(batch_size=batch_size, name='test')
    test_leght = test_manager.len_data

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=train_leght//batch_size,
        decay_rate=0.92, staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    @tf.function
    def train_step(x, y,tran_paras):
        with tf.GradientTape() as tape:
            loss, batch_pred_seq, lens, tran_paras = model(x, y,tran_paras,training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, batch_pred_seq, tran_paras

    # 加载数据集
    train_loss = []
    test_loss = []
    accury = []

    for e in tqdm(range(15)):
        loss_epoch = 0
        start = time.time()

        for i, batch in enumerate(train_manager.iter_batch(shuffle=True)):
            tran_paras = tf.Variable(tf.fill(dims=(label_num, label_num),value=0.5))
            loss_step, pred_step, tran_paras = train_step(batch[:-1], batch[-1],tran_paras)
            loss_epoch += loss_step
            if (i + 1) % 100 == 0:
                print('\n第 %d epoch 的第%d步||||共%d步的train损失是%f' % (e + 1, i + 1, train_leght, loss_step))
        end = time.time()
        print('\n第 %d 个epoch的平均损失是%f||||花费时间%f s' % (e + 1, loss_epoch / train_leght, end - start))
        train_loss.append(loss_epoch / train_leght)
        # 保存模型
        model.save_weights(os.path.join(savepath, 'eckpt'))

        test_loss_epoch = 0
        results = []
        total_num = 0
        total_correct = 0
        # print(":",tran_paras)
        precisions, recalls, f1_score = 0, 0, 0
        for i, batch in enumerate(test_manager.iter_batch(shuffle=True)):
            loss, pred, lengths = model(batch[:-1], batch[-1], tran_paras)
            test_loss_epoch += loss
            chars = batch[0]
            labels = batch[-1]
            for j in range(len(pred)):
                length = lengths[j]
                string = [mapping_dict['word'][0][index] for index in chars[j][:length]]
                label = [mapping_dict['label'][0][index] for index in labels[j][:length]]
                tags = [mapping_dict['label'][0][index] for index in pred[j][:length]]
                precision, recall, f1 = calculatePRF1(label,tags)
                precisions += precision
                recalls += recall
                f1_score += f1
                result = [k for k in zip(string, tags)]
                results.append(result)
                correct = tf.cast(tf.equal(label, tags), dtype=tf.int32)
                correct = tf.reduce_sum(correct)
                total_num += length
                total_correct += int(correct)
            if (i + 1) % 20 == 0:
                print('\n第 %d epoch 的第%d步||||共%d步的test损失是%f||精度为：%f||召回率为：%f||f1_score为：%f'
                      % (e + 1, i + 1, test_leght, loss, precisions/(i+1)/batch_size,
                         recalls/(i+1)/batch_size, f1_score/(i+1)/batch_size))
        acc = total_correct / total_num
        accury.append(acc)
        # print(results[-1])
        test_loss.append((test_loss_epoch / test_leght).numpy())
        print("\n第 %d 个epoch的平均test_loss:%f|||第 %d 个epoch的acc:%f" % (e + 1, test_loss[-1],e + 1, accury[-1]))
        if (len(train_loss) >= 5) and (train_loss[-1] >= train_loss[-2]) and (train_loss[-2] >= train_loss[-3]):
            break

    plt.figure()
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss, color='C0', marker='s', label='训练losses')
    plt.ylabel('y_label')
    plt.xlabel('step')
    plt.legend()
    plt.grid()
    plt.title('simple plot')
    plt.savefig('train_loss1.svg')
    plt.close()
    plt.figure()
    x = [i for i in range(len(accury))]
    plt.plot(x, accury, color='C1', marker='s', label='测试accuracy')
    plt.ylabel('y_label')
    plt.xlabel('step')
    plt.legend()
    plt.grid()
    plt.title('simple plot')
    plt.savefig('test_accuracy1.svg')
    plt.close()
    plt.figure()
    x = [i for i in range(len(test_loss))]
    plt.plot(x, test_loss, color='C0', marker='s', label='训练losses')
    plt.ylabel('y_label')
    plt.xlabel('step')
    plt.legend()
    plt.grid()
    plt.title('simple plot')
    plt.savefig('test_loss1.svg')
    plt.close()


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if __name__ == "__main__":

    batchsz = 20
    hide_dim = 100
    label_num = 31
    attention_dim = 200
    dicts = []
    dicts.append([len(mapping_dict['word'][0]), 100])
    dicts.append([len(mapping_dict['bound'][0]), 20])
    dicts.append([len(mapping_dict['flag'][0]), 50])
    dicts.append([len(mapping_dict['radical'][0]), 50])
    dicts.append([len(mapping_dict['pinyin'][0]), 50])
    train_and_test(batchsz, dicts, hide_dim, label_num, attention_dim)
