import os
import csv
import jieba
import json
from collections import Counter
import pickle
from ltp import LTP
import nltk
ltp = LTP()


class FileContent:
    def __init__(self):
        self.answer_options = []
        self.right_answer = ""
        self.keywords = []

    def add_answer_options(self, option):
        self.answer_options.append(option)

    def set_right_answer(self, answer):
        self.right_answer = answer


def file_content_to_dict(f):
    return {
        'keywords': f.keywords,
        'right answer': f.right_answer,
        'options': f.answer_options,
    }


class PreProcess:
    def __init__(self):
        # 文件路径
        self.initial_train_data = '../data/TrecQA/train.tsv'
        self.initial_validation_data = '../data/TrecQA/dev.tsv'
        self.initial_test_data = '../data/ChineseDBQA/nlpcc2017.dbqa.test'
        self.processed_train = './input/TrecQA_train.json'
        self.processed_val = './input/TrecQA_dev.json'
        self.processed_test = '../data/input/test.json'

        # if os.path.isfile('../data/stopwords.txt'):
        #     fp = open('../data/stopwords.txt', 'r', encoding='utf-8')
        #     self.stopwords = [line.strip('\n') for line in fp.readlines()]
        # else:
        #     raise Exception("stop words file not exists!\n")

    # 数据预处理
    # def read_tsv_file(self, filename):
    #     """
    #     :param filename: string
    #     :return: res: dictionary
    #     """
    #     res = dict()
    #     if os.path.isfile(filename):
    #         tsv_file = open(filename, 'r', encoding="utf-8")
    #         read_tsv = csv.reader(tsv_file, delimiter="\t")
    #         for row in read_tsv:
    #             label, question, content = row
    #             print(question)
    #             if res.get(question) is None:
    #                 res[question] = []
    #             res[question].append((label, self.tokenize(question, ngram=1), self.tokenize(content, ngram=1)))
    #     else:
    #         raise Exception(filename + " not exists\n")
    #
    #     return res

    def read_tsv_file(self, filename, lang="en"):
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
                res[question].append((label, self.tokenize(question, ngram=1, lang="en"), self.tokenize(content, ngram=1, lang="en")))
        else:
            raise Exception(filename + " not exists\n")
        return res

    def tokenize(self, str, ngram=1, lang="en"):
        """
        对句子进行分词。包含去除停止词功能，支持生成n-gram
        :param str: string
        :param ngram: if ngram=2, generates 1-gram and 2-gram
        :return: word_list: list
        """
        if lang == "zh":
            # 分词：精确模式
            word_list = jieba.cut(str, cut_all=False)
        else:
            word_list = nltk.word_tokenize(str)

        # 去除停止词
        # word_list = [word for word in word_list if word not in self.stopwords and not word.isspace()]
        word_list = [word for word in word_list if not word.isspace()]
        if ngram == 1:
            return word_list
        sent_len = len(word_list)
        if sent_len < ngram:
            return word_list
        word_list = ["" if lang == "zh" else " ".join(word_list[index: index + k])
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
        with open('../data/models/word2idx_en.pickle', 'wb') as handle:
            pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def keyword_extraction(self, file_content):
        """
        extract keyword from question using named-entity recognition
        :param file_content: FileContent()
        :return:
        """

        # [question, question....]
        for key, value in file_content.items():
            seg, hidden = ltp.seg([key])
            # ner: [[('Nh', 2, 2)]]
            ner = ltp.ner(hidden)
            # keywords: [('PERSON', "吴轩")],  tuple_item: ('Nh', 2, 2)
            keywords = [seg[0][tuple_item[1]: tuple_item[2] + 1] for tuple_item
                        in ner[0]]
            ngram_keywords = list(map(lambda keyword: self.get_ngram(keyword, ngram=2), keywords))
            # flatten
            ngram_keywords = [item for keywords_list in ngram_keywords for item in keywords_list]
            file_content[key].keywords = ngram_keywords

        return file_content

    def get_ngram(self, sent, ngram):
        """
        输入已经分完词的option, 去除停止词后，输出该option的nGram形式
        :param sent: ['北京科技大学', '，', '简称', '北科', '或',]
        :param ngram: ['北京科技大学','北京科技大学简称','北京科技大学简称北科']
        :return:
        """
        # sent = [word for word in sent if word not in self.stopwords]
        sent = [word for word in sent if not word.isspace()]
        if ngram == 1:
            return sent
        sent_len = len(sent)
        if sent_len < ngram:
            return sent
        ngram_list = ["".join(sent[index: index+k])
                      for k in range(1, ngram + 1) for index in range(0, sent_len - ngram + 1)]
        return ngram_list


if __name__ == "__main__":
    # preprocess = PreProcess()
    # result = preprocess.read_tsv_file(preprocess.initial_train_data)
    # with open(preprocess.processed_train, 'w', encoding="utf-8") as fp:
    #     json.dump(result, fp, indent=4, ensure_ascii=False)
    # preprocess.get_word2idx()
    # with open("../data/output/fileContent.json", 'r', encoding="utf-8") as load_j:
    #     file_content = json.load(load_j)
    # preprocess = PreProcess()
    # preprocess.keyword_extraction(file_content)

    pre_process = PreProcess()
    pre_process.get_word2idx()
    # train_data = pre_process.read_tsv_file(pre_process.initial_validation_data, lang="en")
    # # train_data = pre_process.keyword_extraction(train_data)
    # with open(pre_process.processed_val, 'w', encoding="utf-8") as fp:
    #     json.dump(train_data, fp, indent=2, ensure_ascii=False, default=file_content_to_dict)

