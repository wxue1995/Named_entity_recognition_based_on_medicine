import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm  #列表能够使用
import os
import math,random


#列表拼接

def joint_list(results, lists):
    for data in lists:
        results.append(data)
    return results


def item2id(data, w2i):
    return [w2i[x] if x in w2i else w2i["UNK"] for x in data]


def get_data_windows(name='train'):
    with open(f'data/prepare/dict.pkl','rb') as f:
        map_dict = pickle.load(f)

    results = []
    root = os.path.join('data/prepare/', name)
    files = os.listdir('data/prepare/'+name)
    for file in tqdm(files):

        result = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path, sep=",")
        # print(samples)
        length = len(samples)
        sep_index = samples[samples["word"] == 'sep'].index.tolist()#拿到分割行的下标
        sep_index = list(sorted(set(sep_index+[-1,length])))
        # print(sep_index,len(sep_index)-1)
        for i in range(len(sep_index)-1):

            start = sep_index[i] + 1
            end = sep_index[i+1] - 1
            data = []
            for feature in samples.columns:#访问每一列
                # print(map_dict[feature][1])  #w2id
                data.append(item2id(samples[feature][start:end], map_dict[feature][1]))
            result.append(data)

        # print("row:",np.array(result).shape)
        # # # print(result, len(result))  # 对应每句话的word,label,flag.....的id

        #----------------------------数据增强——————————————————————————————
        joint_two = []
        for i in range(len(result)-1):
            # if len(result[i][0]) < 15:
                # print("+++++++++++",result[i][1],result[i+1][1])
            joint_two.append([result[i][j]+result[i+1][j] for j in range(len(result[i]))])
        # print("joint_two:",np.array(joint_two).shape)

        #
        joint_three = []
        for i in range(len(result) - 2):
            # if len(result[i][0]) < 15:
                # print("+++++++++++",result[i][1],result[i+1][1],result[i+2][1])
            joint_three.append([result[i][j] + result[i + 1][j]+result[i+2][j]
                                    for j in range(len(result[i]))])
        # print("joint_three:",np.array(joint_three).shape)
        # results = joint_list(results, result)
        # results = joint_list(results, joint_two)
        # results = joint_list(results, joint_three)
        results.extend(result+joint_two+joint_three)
    print("all_shape:",np.array(results).shape)

    with open(f'data/prepare/'+name+'.pkl','wb') as f:
        pickle.dump(results,f)


def get_dict(path):
    with open(path,'rb') as f:
        dict=pickle.load(f)
    return dict


class BatchManager(object):

    def __init__(self, batch_size, name='train', predict=None):
        with open(f'data/prepare/'+name+'.pkl', 'rb') as f:
            data = pickle.load(f)
        if predict:
            self.batch_data = self.sort_and_pad(data, batch_size, predict)
            self.len_data = len(self.batch_data)
        else:
            self.batch_data = self.sort_and_pad(data[:1000], batch_size)
            self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size, predict=None):
        num_batch = int(math.ceil(len(data) / batch_size))
        if predict:
            batch_data = self.pad_data(data, predict)
            return batch_data
        else:
            sorted_data = sorted(data, key=lambda x: len(x[0]))
            batch_data = []
            for i in range(num_batch):
                batch_data.append(self.pad_data(sorted_data[i*int(batch_size): (i+1)*int(batch_size)], predict))
            return batch_data

    @staticmethod
    def pad_data(data, predict=None):
        chars = []
        labels = []
        bounds = []
        flags = []
        radicals = []
        pinyins = []
        max_length = max([len(sentence[0]) for sentence in data])  #len(data[-1][0])
        for line in data:
            if predict:
                char, bound, flag, radical, pinyin = line
                label = [0] * len(char)  ###预测时不需要标签，为了符合程序，所以造了一个label
            else:
                char, label, bound, flag, radical, pinyin = line
            padding = [0] * (max_length - len(bound))
            chars.append(char + padding)
            labels.append(label + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)

        return [chars, bounds, flags, radicals, pinyins, labels]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]








if __name__ == '__main__':
    get_data_windows(name='train')
    get_data_windows(name='test')
    train_data = BatchManager(10,"train")
    dict = train_data.batch_data
    for batch in train_data.iter_batch( shuffle=True):
        print(len(batch[0]))
        break
    # print(dict)