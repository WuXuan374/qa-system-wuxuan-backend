from nltk.corpus import stopwords
import nltk
import os
import json
import string
import numpy as np
from numpy import dot
from numpy.linalg import norm


class EmbeddingRetrieval:
    def __init__(self, options, word_vector=None, ngram=1, lang="en"):
        self.options = options
        self.lang = lang
        if self.lang == "zh" and os.path.isfile('./data/stopwords.txt'):
            fp = open('./data/stopwords.txt', 'r', encoding='utf-8')
            self.stopwords = [line.strip('\n') for line in fp.readlines()]
        elif self.lang == "en":
            self.stopwords = stopwords.words("english")

        self.ngram_options = list(map(lambda option: self.get_ngram(option, ngram=ngram), self.options))

        self.word_vector = word_vector

        self.word_embedding = list(map(lambda option: np.array([self.word_vector[w] if w in self.word_vector else np.zeros((300,), dtype=float) for w in option]), self.ngram_options))

    def get_ngram(self, sent, ngram):
        """
        输入已经分完词的option, 去除停止词后，输出该option的nGram形式
        :param sent: ['北京科技大学', '，', '简称', '北科', '或',]
        :param ngram: ['北京科技大学','北京科技大学简称','北京科技大学简称北科']
        :return:
        """
        # sent = [word for word in sent if word not in self.stopwords and not word.isspace()]
        # sent = [word for word in sent if word not in string.punctuation]
        # sent = [nltk.WordNetLemmatizer().lemmatize(word) for word in sent]
        sent = [word for word in sent if word not in self.stopwords]
        if ngram == 1:
            return sent
        sent_len = len(sent)
        if sent_len < ngram:
            return sent
        ngram_list = [("" if self.lang == "zh" else " ").join(sent[index: index+k])
                      for k in range(1, ngram + 1) for index in range(0, sent_len - ngram + 1)]
        return ngram_list

    def get_similar_option(self, question_vector):
        """
        get top3 similar answer candidate by compute_similarity between query and answer candidate
        :param question_vector: 2D array, e.g. [[0.7, 0,9],]
        :return: top_options: 对options按照cosine similarity 进行排序
        """
        top_options = []
        for index in range(0, len(self.word_embedding)):
            if len(self.word_embedding[index].shape) < 2 or self.word_embedding[index].shape[1] != 300:
                print(self.options[index])
                continue
            similarity = sum(sum(dot(self.word_embedding[index], question_vector.T)\
                         /(norm(self.word_embedding[index])* norm(question_vector))))
            # 过滤掉similarity score <= 0 的候选答案
            if similarity > 0:
                top_options.append((index, similarity))
        length = len(top_options)
        if length == 0:
            return [None]
        return sorted(top_options, key=lambda x: x[1], reverse=True)[:3 if length >= 3 else length]

    def query(self, question_embedding_vector):
        """
        input a question vector, return top3 relevant answers
        :param question_embedding_vector: 2D array, e.g. [[0.7, 0,9],]
        :return: possible_answers: list of object,  [{"answer": 76个本科专业", "score":0.1555555}]
        """
        answers = self.get_similar_option(question_embedding_vector)
        possible_answers = []
        if answers == [None]:
            return possible_answers
        for index, sim in answers:
            answer_str = ("" if self.lang == "zh" else " ").join(self.options[index])
            print({"answer": answer_str, "first_score": sim})
            possible_answers.append(
                {"answer": answer_str, "first_score": sim})
        return possible_answers

if __name__ == '__main__':
    # Convert
    # input_file = './data/word2vec/glove.6B.300d.txt'
    # output_file = './data/word2vec/gensim_glove.6B.300d.txt'
    # glove2word2vec(input_file, output_file)

    sourcefile = './data/TFIDF_input/TrecQA_train.json'
    with open(sourcefile, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
    options = content["what are the valdez principles ?"]["options"]
    eRetrieval = EmbeddingRetrieval(options)



    # model = KeyedVectors.load_word2vec_format('./data/word2vec/gensim_glove.6B.300d.txt', binary=False)
    # word_vectors = model.wv
    # word_vectors.save('./data/word2vec/vectors.kv')
    # reloaded_word_vectors = KeyedVectors.load('./data/word2vec/vectors.kv')
    # print(reloaded_word_vectors.most_similar('cat'))
