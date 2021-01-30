import numpy as np
import os
import json
from process_question import ProcessQuestion
from extractor import extract_date
from ltp import LTP
from numpy import dot
from numpy.linalg import norm
import gensim
ltp = LTP()


def word2vec(tokens, embeddings):
    """
    输入词链表，生成对应的word embedding vector
    :param tokens: list, e.g, ["浙江大学", "人文学院"]
    :param embeddings: gensim 读取预训练词向量的结果
    :return: word_vec: list of array, e.g. [[0.1,0.3...,0.5]], 每一个数组是300维的
    """
    dim = embeddings["和"].size
    word_vec = []
    for word in tokens:
        if word in embeddings and embeddings[word].size == dim:
            word_vec.append(embeddings[word])
        else:
            word_vec.append(np.random.uniform(-0.25, 0.25, dim))
    # no word in tokens in given, word_vec is still []
    if not word_vec:
        word_vec.append(np.random.uniform(-0.25, 0.25, dim))
    return np.array(word_vec, ndmin=2)


class RetrievalWordEmbedding:
    def __init__(self, options, answer_types, embeddings, ngram=1):
        # 答案的预期类型，根据问题得到的
        self.answer_types = answer_types
        # 首先根据问句类型，对候选答案进行一次筛选
        self.options = list(filter
                            (lambda option: self.filter_options_by_answer_type(self.answer_types, option), options))
        if os.path.isfile('./data/stopwords.txt'):
            fp = open('./data/stopwords.txt', 'r', encoding='utf-8')
            self.stopwords = [line.strip('\n') for line in fp.readlines()]
        else:
            raise Exception("stop words file not exists!\n")
        self.ngram_options = list(map(lambda option: self.get_ngram(option, ngram=ngram), self.options))
        # list of 2D array
        self.embedding_vectors = list(map(lambda option: word2vec(option, embeddings), self.ngram_options))

    def filter_options_by_answer_type(self, answer_types, option):
        """
        利用之前标记的answer types来对候选答案进行预处理
        如果候选答案中不含answer types中所要求的内容，则派出该候选答案
        :param answer_types: list, e.g. ["PERSON", "LOCATION"]
        :param option: list, tokenized answer, ["重庆市委","副","书记","邢元敏"],
        :return:
        """
        if not answer_types:
            return True
        answer_str = "".join(option)
        named_entities = self.get_named_entity(answer_str)
        for entity in named_entities:
            if entity[0] == 'Nh' and "PERSON" in answer_types:
                return True
            elif entity[0] == 'Ns' and "LOCATION" in answer_types:
                return True
            elif entity[0] == 'Ni' and "ORGANIZATION" in answer_types:
                return True
            elif "DATE" in answer_types and extract_date(answer_str):
                return True
        return False

    def get_ngram(self, sent, ngram):
        """
        输入已经分完词的option, 去除停止词后，输出该option的nGram形式
        :param sent: ['北京科技大学', '，', '简称', '北科', '或',]
        :param ngram: ['北京科技大学','北京科技大学简称','北京科技大学简称北科']
        :return:
        """
        sent = [word for word in sent if word not in self.stopwords]
        if ngram == 1:
            return sent
        sent_len = len(sent)
        if sent_len < ngram:
            return sent
        ngram_list = ["".join(sent[index: index+k])
                      for k in range(1, ngram + 1) for index in range(0, sent_len - ngram + 1)]
        return ngram_list

    def get_named_entity(self, question_str):
        """
        从候选答案中识别命名实体
        :param question_str: str, e.g. "学校位于北京“
        :return: named_entities: list, e.g. ['Nh', "吴轩"]
        """
        seg, hidden = ltp.seg([question_str])
        # ner: [[('Nh', 2, 2)]]
        ner = ltp.ner(hidden)
        # keywords: [('PERSON', "吴轩")],  tuple_item: ('Nh', 2, 2)
        named_entities = [(tuple_item[0], "".join(seg[0][tuple_item[1]: tuple_item[2] + 1]))
                          for tuple_item in ner[0]]
        return named_entities

    def get_concrete_by_answer_type(self, named_entities, answer_types, answer_str):
        """
        利用之前标注好的预期答案类型，从options中抽取更加精确的答案
        :param named_entities: list, e.g. [ "Nh", "吴轩"]
        :param answer_types: list, e.g. ["PERSON", "LOCATION"]
        :param answer_str: string
        :return: concrete_answers: list
        """
        concrete_answers = []
        if not answer_types:
            return concrete_answers
        for type in answer_types:
            # 以下三类：直接通过命名实体识别技术，找到相应命名实体
            if type == "PERSON":
                for entity in named_entities:
                    if entity[0] == 'Nh':
                        concrete_answers.append(entity[1])
                        break
            elif type == "LOCATION":
                for entity in named_entities:
                    if entity[0] == 'Ns':
                        concrete_answers.append(entity[1])
                        break
            elif type == "ORGANIZATION":
                for entity in named_entities:
                    if entity[0] == 'Ni':
                        concrete_answers.append(entity[1])
                        break
            elif type == "DATE":
                dates = extract_date(answer_str)
                if dates:
                    concrete_answers.append(",".join(dates))
                break

        return concrete_answers

    def get_similar_option(self, question_vector):
        """
        get top3 similar answer candidate by compute_similarity between query and answer candidate
        :param embedding_vector: 2D array, e.g. [[0.7, 0,9],]
        :return: top_options: 对options按照cosine similarity 进行排序
        """
        top_options = []
        for index in range(0, len(self.embedding_vectors)):
            if self.embedding_vectors[index].shape[1] != 300:
                print(self.options[index])
            similarity = sum(sum(dot(self.embedding_vectors[index], question_vector.T)\
                         /(norm(self.embedding_vectors[index])*norm(question_vector))))
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
            # possible_answers.append(("".join(self.options[index]), sim))
            answer_str = "".join(self.options[index])
            named_entities = self.get_named_entity(answer_str)
            concrete_answers = self.get_concrete_by_answer_type(named_entities, self.answer_types, answer_str)
            if concrete_answers:
                print({"answer": answer_str, "score": sim, "concrete_answer": ", ".join(concrete_answers)})
            possible_answers.append({"answer": answer_str, "score": sim, "concrete_answer": ", ".join(concrete_answers)})
        return possible_answers


if __name__ == '__main__':
    sourcefile = './data/output/fileContent.json'
    with open(sourcefile, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
    embedding_file = "./data/word2vec/word2vec-300.iter5"
    print("embedding reading")
    embeddings = gensim.models.KeyedVectors. \
        load_word2vec_format(embedding_file, binary=False, unicode_errors='ignore')
    question = ProcessQuestion("重庆市兼善中学是谁、在哪一年建立的？", "./data/stopwords.txt", embeddings, ngram=2)
    question_embedding = question.question_embedding
    answers = RetrievalWordEmbedding(content["重庆市兼善中学是谁、在哪一年建立的？"]["options"],
                                     question.answer_types, embeddings, ngram=2)
    possible_answers = answers.query(question_embedding)
    print(possible_answers)
