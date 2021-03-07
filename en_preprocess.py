import os
import csv
import jieba
import json
from collections import Counter
import pickle
import nltk
from ltp import LTP
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
        self.initial_train_data = './data/wikiQA/WikiQA-train.tsv'
        self.initial_validation_data = '../data/ChineseDBQA/nlpcc2017.dbqa.dev'
        self.initial_test_data = '../data/ChineseDBQA/nlpcc2017.dbqa.test'
        self.processed_train = './data/TFIDF_input/WikiQA_train.json'
        self.processed_val = '../data/TFIDF_input/validation.json'
        self.processed_test = '../data/input/test.json'

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
                _, question, _, keywords, _, content, label = row
                print(question, keywords, content, label)
                if res.get(question) is None:
                    res[question] = FileContent()
                res[question].add_answer_options(self.tokenize(content, ngram=1))
                res[question].keywords = keywords
                if label == '1':
                    res[question].set_right_answer(self.tokenize(content, ngram=1))
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
        word_list = [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(str)]

        if ngram == 1:
            return word_list
        sent_len = len(word_list)
        if sent_len < ngram:
            return word_list
        word_list = ["".join(word_list[index: index + k])
                     for k in range(1, ngram + 1) for index in range(0, sent_len - ngram + 1)]
        return word_list


if __name__ == "__main__":
    pre_process = PreProcess()
    train_data = pre_process.read_tsv_file(pre_process.initial_train_data)
    with open(pre_process.processed_train, 'w', encoding="utf-8") as fp:
        json.dump(train_data, fp, indent=2, ensure_ascii=False, default=file_content_to_dict)
