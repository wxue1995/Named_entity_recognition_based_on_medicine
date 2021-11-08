import tensorflow as tf
from data_processs import split_text
from predict_data_process import predict_process_word
from model import Model_NER
from data_utils import get_dict
import os


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if __name__ == "__main__":
    savepath = "checkpoints/"
    dict_file = 'data/prepare/dict.pkl'
    mapping_dict = get_dict(dict_file)
    dicts = []
    dicts.append([len(mapping_dict['word'][0]), 100])
    dicts.append([len(mapping_dict['bound'][0]), 20])
    dicts.append([len(mapping_dict['flag'][0]), 50])
    dicts.append([len(mapping_dict['radical'][0]), 50])
    dicts.append([len(mapping_dict['pinyin'][0]), 50])

    model = Model_NER(dicts, hide_dim=100, label_num=31,attention_size=200)
    model.load_weights(os.path.join(savepath, 'ckpt'))
    print('load save_model weight----over!')
    inputs = input("请输入需要查询的话：")
    results = split_text(inputs)
    print(len(results))
    batch = predict_process_word(results)
    _, pred, lengths,_ = model(batch[:-1], batch[-1], model.variables[-1], training=True)
    print(pred,len(pred),lengths)
    chars = batch[0]
    print(chars)
    data = []
    for j in range(len(pred)):
        length = lengths[j]
        string = [mapping_dict['word'][0][index] for index in chars[j][:length]]
        tags = [mapping_dict['label'][0][index] for index in pred[j][:length]]
        result = [k for k in zip(string, tags)]
        data.append(result)
    print(data)
    print(results)
