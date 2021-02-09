import os
import csv
import jieba
import json
from collections import Counter
import pickle


class PreProcess:
    def __init__(self):
        # 文件路径
        self.initial_train_data = '../data/ChineseDBQA/nlpcc2016.dbqa.test'
        self.initial_validation_data = '../data/ChineseDBQA/nlpcc2017.dbqa.dev'
        self.initial_test_data = '../data/ChineseDBQA/nlpcc2017.dbqa.test'
        self.processed_train = '../data/input/train.json'
        self.processed_val = '../data/input/validation.json'
        self.processed_test = '../data/input/test.json'

        if os.path.isfile('../data/stopwords.txt'):
            fp = open('../data/stopwords.txt', 'r', encoding='utf-8')
            self.stopwords = [line.strip('\n') for line in fp.readlines()]
        else:
            raise Exception("stop words file not exists!\n")

    # 数据预处理
    def read_tsv_file(self, filename):
        """
        :param filename: string
        :return: res: dictionary
        """
        res = dict()
        if os.path.isfile(filename):
            tsv_file = open(filename, 'r', encoding="utf-8")
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for row in read_tsv:
                label, question, content = row
                print(question)
                if res.get(question) is None:
                    res[question] = []
                res[question].append((label, self.tokenize(question, ngram=1), self.tokenize(content, ngram=1)))
        else:
            raise Exception(filename + " not exists\n")

        return res

    def tokenize(self, str, ngram=1):
        """
        对句子进行分词。包含去除停止词功能，支持生成n-gram
        :param str: string
        :param ngram: if ngram=2, generates 1-gram and 2-gram
        :return: word_list: list
        """
        # 分词：精确模式
        word_list = jieba.cut(str, cut_all=False)
        # 去除停止词
        word_list = [word for word in word_list if word not in self.stopwords and not word.isspace()]
        if ngram == 1:
            return word_list
        sent_len = len(word_list)
        if sent_len < ngram:
            return word_list
        word_list = ["".join(word_list[index: index + k])
                     for k in range(1, ngram + 1) for index in range(0, sent_len - ngram + 1)]
        return word_list

    def get_word2idx(self):
        dict_size = 70000
        with open(self.processed_train, 'r', encoding="utf-8") as load_j:
            train_data = json.load(load_j)
        all_words = []
        for question in train_data.keys():
            for item in train_data[question]:
                all_words.extend(item[1])
                all_words.extend(item[2])
        freq_dict = Counter(all_words)
        idx2word = dict(enumerate([word for (word, freq) in freq_dict.most_common(dict_size)]))
        word2idx = dict([(value, key) for (key, value) in idx2word.items()])
        with open('../data/models/word2idx.pickle', 'wb') as handle:
            pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    preprocess = PreProcess()
    result = preprocess.read_tsv_file(preprocess.initial_train_data)
    with open(preprocess.processed_train, 'w', encoding="utf-8") as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)
    # preprocess.get_word2idx()
