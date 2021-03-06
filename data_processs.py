import os
import re
import numpy as np
local = os.getcwd()
locals = local + "\\datas\\ruijin_round1_train2_20181022"
file_name_list = os.listdir(locals)
files = list(set([file.split('.')[0] for file in file_name_list]))


##统计各个实体的个数
def get_entities(loc):
    entities = {}
    for i, file in enumerate(files):
        path = os.path.join(loc, files[i] + '.ann')
        with open(path,'r',encoding='utf-8') as f:
            text = f.readlines()
            for line in text:
                name = line.split("\t")[1].split(' ')[0]
                if name in entities:
                    entities[name] += 1
                else:
                    entities[name] = 1

    return entities


# 给实体打上标签
def get_label_encoder(entities):
    entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
    entities = [x[0] for x in entities]
    id2label = []
    id2label.append("O")
    for entity in entities:
        id2label.append("B-"+entity)
        id2label.append("I-"+entity)
    label2id = {id2label[i]:i for i in range(len(id2label))}
    return id2label,label2id


##判断是否是中文
def ischinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


# 清理数据后断句
def split_text(text):
    split_index = []
    pattern1 = '。|，|,|;|；|\.|\?'  ###.和？都需要用\进行转义

    for m in re.finditer(pattern1, text):
        # print(m.group())
        idx = m.span()[0]
        # print(m.span())
        if text[idx - 1] == '\n':
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isdigit():  # 前后是数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isspace() and text[idx + 2].isdigit():  # 前数字 后空格 后后数字
            continue
        if text[idx - 1].islower() and text[idx + 1].islower():  # 前小写字母后小写字母
            continue
        if text[idx - 1].islower() and text[idx + 1].isdigit():  # 前小写字母后数字
            continue
        if text[idx - 1].isupper() and text[idx + 1].isdigit():  # 前大写字母后数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].islower():  # 前数字后小写字母
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isupper():  # 前数字后大写字母
            continue
        if m.group() =="。" and idx != len(text)-1:
            if text[idx:idx+1] == "。" and text[idx+1].isdigit():
                split_index.append(idx+1)
        if text[idx - 1].isspace() and text[idx - 2].isspace() and text[idx - 3] == 'C':  # HBA1C的问题
            continue
        if text[idx - 1].isspace() and text[idx - 2] == 'C':
            continue
        if text[idx - 1].isupper() and text[idx + 1].isupper():  # 前大些后大写
            continue
        if text[idx] == '.' and text[idx + 1:idx + 4] == 'com':  # 域名
            continue
        split_index.append(idx + 1)
        if idx != len(text):
            if text[idx] in set('.。;；,，'):  # 前句号后句号
                continue
        elif idx == len(text) - 1:
            split_index.append(idx+1)
    pattern2 = '\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'
    pattern2 += '注:|附录 |表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,;]+?\n|图 \d|Fig \d|'
    pattern2 += '\[Abstract\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论|'
    pattern2 += 'and |or |with |by |because of |as well as '
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if (text[idx:idx + 2] in ['or', 'by'] or text[idx:idx + 3] == 'and' or text[idx:idx + 4] == 'with') \
                and (text[idx - 1].islower() or text[idx - 1].isupper()):
            continue
        split_index.append(idx)

    pattern2 += '\n\d\.'  # 匹配1.  2.  这些序号
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if ischinese(text[idx + 3]):
            split_index.append(idx + 1)

    for m in re.finditer('\n\(\d\)', text):  # 匹配(1) (2)这样的序号
        idx = m.span()[0]
        split_index.append(idx + 1)
    split_index = list(sorted(set([0, len(text)] + split_index)))

    other_index = []
    for i in range(len(split_index) - 1):
        begin = split_index[i]
        end = split_index[i + 1]
        if text[begin] in '一二三四五六七八九零十' or \
                (text[begin] == '(' and text[begin + 1] in '一二三四五六七八九零十'):  # 如果是一、和(一)这样的标号
            for j in range(begin, end):
                if text[j] == '\n':
                    other_index.append(j + 1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    other_index = []
    for i in range(len(split_index) - 1):  # 对长句子进行拆分
        b = split_index[i]
        e = split_index[i + 1]
        other_index.append(b)
        if e - b > 150:
            for j in range(b, e):
                if (j + 1 - other_index[-1]) > 15:  # 保证句子长度在15以上
                    if text[j] == '\n':
                        other_index.append(j + 1)
                    if text[j] == ' ' and text[j - 1].isnumeric() and text[j + 1].isnumeric():
                        other_index.append(j + 1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    for i in range(1, len(split_index) - 1):  # 10   20  干掉全部是空格的句子
        idx = split_index[i]
        while idx > split_index[i - 1] - 1 and text[idx - 1].isspace():
            idx -= 1
        split_index[i] = idx
    split_index = list(sorted(set([0, len(text)] + split_index)))


    # 处理短句子
    temp_idx = []
    i = 0
    while i < len(split_index) - 1:  # 0 10 20 30 45
        b = split_index[i]
        e = split_index[i + 1]

        num_ch = 0
        num_en = 0
        if e - b < 15:
            for ch in text[b:e]:
                if ischinese(ch):
                    num_ch += 1
                elif ch.islower() or ch.isupper():
                    num_en += 1
                # else:
                #     split_index.pop(e)
                #     break
                if num_ch + 0.5 * num_en > 5:  # 如果汉字加英文超过5个  则单独成为句子
                    temp_idx.append(b)
                    i += 1
                    break
            if num_ch + 0.5 * num_en <= 5 and e != split_index[-1] and e != split_index[-2]:  # 如果汉字加英文不到5个  和前面一个句子合并
                # temp_idx.append(b)
                i += 2
            elif num_ch + 0.5 * num_en <= 5:
                i +=2

        else:
            temp_idx.append(b)
            i += 1
    split_index = list(sorted(set([0, len(text)] + temp_idx)))
    #测试长短句子
    q = []
    for i in range(len(split_index) - 1):
        s = split_index[i]
        e = split_index[i + 1]
        leng = e - s
        q.append(leng)
    # x = text[split_index[np.argmin(q)]:np.argmin(q) + 1]
    # print(text[split_index[np.argmin(q)]-20:split_index[np.argmin(q)+1]+20])
    # print("+++++++", max(q), min(q),split_index[np.argmin(q)],split_index[np.argmin(q)+1])

    result = []
    for i in range(len(split_index) - 1):
        result.append(text[split_index[i]:split_index[i + 1]])
    # print(result)

    # 做一个检查
    s = ''
    for r in result:
        s += r
    assert len(s) == len(text)
    return result





if __name__ == '__main__':
    entities = get_entities(locals)
    id2label,label2id = get_label_encoder(entities)
    # print(id2label, label2id)
    results = []
    for i, file in enumerate(files):
        path = os.path.join(locals, files[i] + '.txt')
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            result = split_text(text)
            print(result)
            break




    # file_name_list = os.listdir(locals)
    # files = list(set([file.split('.')[0] for file in file_name_list]))
    # for i, file in enumerate(files):
    #     path = os.path.join(locals, files[i] + '.txt')
    #     with open(path, 'r', encoding='utf-8') as f:
    #         words = f.read()
    #         obj = re.compile(r'。|，|,|; |；|\.', re.S)
    #         for word in obj.finditer(words):
    #             start = word.span()[0] - 5
    #             end = word.span()[1] + 5
    #             print('****', words[start:end], '*****')
    #             break
    #

