import jieba.posseg as psg
from cnradical import Radical,RunOption  #获取中文的偏旁和拼音
import jieba
import pickle
from data_utils import BatchManager


def predict_process_word(texts):
    batchsz = len(texts)
    data = {}
    dict1 = '医学词汇大全.txt'
    dict2 = "中外医药名大全.txt"
    dict3 = "疾病名称.txt"
    jieba.load_userdict(dict1)
    jieba.load_userdict(dict2)
    jieba.load_userdict(dict3)
    with open(f'data/prepare/dict.pkl','rb') as f:
        map_dict = pickle.load(f) #获取二进制数据字典

    # ---------------------------------获取标签----------------------------------
    words = []
    for text in texts:
        word = [x for x in text]
        words.append(word)
    data["word"] = words
    # -----------------------------提取词性和词边界特征----------------------------------
    word_bounds = ['M' for s in texts for x in s]  # 首先给所有的字都表上M标记
    word_flags = []  # 用来保存每个字的词性特征
    for text in texts:
        for word, flag in psg.cut(text):
            if len(word) == 1:  # 判断是一个字的词
                start = len(word_flags)  # 拿到起始下标
                word_bounds[start] = 'S'  # 标记修改为S
                word_flags.append(flag)  # 将当前词的词性名加入到wordflags列表
            else:
                start = len(word_flags)  # 获取起始下标
                word_bounds[start] = 'B'  # 第一个字打上B
                word_flags += [flag] * len(word)  # 将这个词的每个字都加上词性标记
                end = len(word_flags) - 1  # 拿到这个词的最后一个字的下标
                word_bounds[end] = 'E'  # 将最后一个字打上E标记

    # --------------------------------------统一截断---------------------------------------
    bounds = []
    flags = []
    start = 0
    end = 0
    for s in texts:
        l = len(s)
        end += l
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        start += l
    data['bound'] = bounds
    data['flag'] = flags

    # ----------------------------------------获取拼音和偏旁特征-------------------------------------
    radical = Radical(RunOption.Radical)  # 提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)  # 用来提取拼音
    # 提取偏旁部首特征  对于没有偏旁部首的字标上PAD
    data['radical'] = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    # 提取拼音特征  对于没有拼音的字标上PAD
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    # print(data)

    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i["UNK"] for x in data]

    #——————————————————————————————————————转换为词向量——————————————————————————————————————————

    lists = ["word", "bound", "flag", "radical", "pinyin"]
    dataset = []
    for i in range(len(data["word"])):
        results = []
        for feature in lists:
            result = item2id(data[feature][i], map_dict[feature][1])
            results.append(result)
        dataset.append(results)
    # print(dataset)
    # print(len(dataset[0][0]),len(dataset[0][1]),len(dataset[0][2]),len(dataset[0][3]),len(dataset[0][4]))

    with open(f'data/prepare/' + "predict" + '.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    #———————————————————————————————————————数据padding————————————————————————————————————

    predict_data = BatchManager(batchsz,name="predict",predict=True)
    batch = predict_data.batch_data
    return batch

