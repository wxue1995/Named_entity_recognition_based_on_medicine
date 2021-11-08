import re


def calculatePRF1(real_list,pre_list):  #计算实体
    precision = -1.0
    recall = -1.0
    f1 = 1.0
    allLocPre = 0  # 所有被模型识别为地名的数量
    locPreCorr = 0  # 被模型识别为地名且正确的数量
    real_list = " ".join(real_list)
    # print("===",real_list,pre_list)
    all_real_label_list = []
    counts = []
    object = re.compile(r'B.*?(?=B)|B.*?(\n)', re.S)
    for i in object.finditer(real_list):
        temp = i.group().strip()
        x = temp.split(' ')
        all_real_label_list.append(x)
        count = 0
        for j in range(len(x)):
            if x[j][-3:] == x[0][-3:]:
                count += 1
        counts.append(count)
    object1 = re.compile(r'.([^B]*)$', re.S)  # 匹配最后一次出现字符B且后面的字符串
    for m in object1.finditer(real_list):
        tmp = m.group().strip()
        y = tmp.split(" ")
        all_real_label_list.append(y)
        c = 0
        for j in range(len(y)):
            if y[j][-3:] == y[0][-3:]:
                c += 1
        counts.append(c)
    # print(counts)
    # print("all_real_label_list:  ",all_real_label_list)
    start, end = 0, 0
    all_pre_label_list = []
    for i in range(len(all_real_label_list)):
        end += len(all_real_label_list[i])
        all_pre_label_list.append(pre_list[start:end])
        start += len(all_real_label_list[i])
    # print("all_pre_label_list:  ",all_pre_label_list)
    allLocReal = len(all_real_label_list)  # 所有地名的数量
    for pre, real, n in zip(all_pre_label_list, all_real_label_list, counts):
        s = 0
        # print(real, pre, n)
        if real[0:n] == pre[0:n]:
            locPreCorr += 1
        for w in real[0:n]:
            if w in pre[0:n]:
                s = 1
        allLocPre += s
    if allLocReal != 0:
        recall = locPreCorr * 1.0 / allLocReal  # 召回率
    if allLocPre != 0:
        precision = locPreCorr * 1.0 / allLocPre  # 查准率
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)  # 调和平均
    return abs(precision), abs(recall), f1




