import jieba
import math
import nltk
from nltk.corpus import stopwords


class Reorder:
    def __init__(self, lang="zh"):
        self.lang = lang

    def compute_score(self, question, answer, ngrams=(1, 3)):
        """
        参考严德美论文P32
        计算候选答案的重排序得分
        :param question: str, 分词后的question
        :param answer: str, 分词后的answer
        :param ngrams: tuple, 所考虑的ngram范围 e.g. (1,3) 代表考虑1-gram 和 2-gram
        :return: m_score: num
        """
        tokenized_question = self.tokenize(question)
        tokenized_answer = self.tokenize(answer)
        m_score = 0.
        for ngram in range(ngrams[0], ngrams[1]):
            ngram_question = self.get_ngram(tokenized_question, ngram)
            ngram_answer = self.get_ngram(tokenized_answer, ngram)
            common = len([ngram for ngram in ngram_question if ngram in ngram_answer])
            p_score = common / len(ngram_question)
            # 0.01 避免p_score为0
            m_score += math.log(p_score + 0.01)
        m_score = 1 / (ngrams[1] - ngrams[0]) * m_score
        return m_score

    def get_ngram(self, tokenized_question, ngram):
        """
        输入已经分完词的question, 去除停止词后，输出该question的nGram形式
        :param tokenized_question: ['北京科技大学', '，', '简称', '北科', '或',]
        :param ngram: num
        :return: ngram_list: ['北京科技大学','北京科技大学简称','北京科技大学简称北科']
        """
        if ngram == 1:
            return tokenized_question
        question_len = len(tokenized_question)
        if question_len < ngram:
            return tokenized_question
        ngram_list = [("" if self.lang == "zh" else " ").join(tokenized_question[index: index+k])
                      for k in range(1, ngram + 1) for index in range(0, question_len - ngram + 1)]
        return ngram_list

    def tokenize(self, str, ngram=1):
        """
        对句子进行分词。包含去除停止词功能，支持生成n-gram
        :param str: string
        :param ngram: if ngram=2, generates 1-gram and 2-gram
        :return: word_list: list
        """
        # 分词：精确模式
        if self.lang == "zh":
            word_list = jieba.cut(str, cut_all=False)
            word_list = [word for word in word_list if not word.isspace()]

        else:
            word_list = nltk.word_tokenize(str)
            stop_words = stopwords.words("english")
            word_list = [word for word in word_list if word not in stop_words]

        if ngram == 1:
            return word_list
        sent_len = len(word_list)
        if sent_len < ngram:
            return word_list
        word_list = [("" if self.lang == "zh" else " ").join(word_list[index: index + k])
                     for k in range(1, ngram + 1) for index in range(0, sent_len - ngram + 1)]

        return word_list

