import os
import json
import math
from process_question import ProcessQuestion
from extractor import extract_date
from ltp import LTP
ltp = LTP()


class RetrievalTFIDF:
    def __init__(self, options, ngram=1):
        self.idf = {}  # dict
        self.optionsInfo = {}
        self.options = options
        if os.path.isfile('./data/stopwords.txt'):
            fp = open('./data/stopwords.txt', 'r', encoding='utf-8')
            self.stopwords = [line.strip('\n') for line in fp.readlines()]
        else:
            raise Exception("stop words file not exists!\n")
        self.ngram_options = list(map(lambda option: self.get_ngram(option, ngram=ngram), self.options))
        self.computeTFIDF()

    def getTFCount(self, option):
        """
        给出一个答案候选，计算该答案中各单词的TF值
        计算公式：该单词的出现次数 / 答案中的单词总数
        :param option: list, 分词之后的候选答案: ['兼善中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区毛背沱', '
        :return: word_frequency: dict, word: term frequency(TF)
        """
        word_frequency = {}
        for word in option:
            if word in self.stopwords:
                continue
            if word in word_frequency.keys():
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        return word_frequency

    def computeTFIDF(self):
        """
        计算各option的单词在语料库中的Inverse Document Frequency(IDF)
        这里的语料库指的是该问题对应的所有候选答案
        Update 01/27: 输入的选项改为取了ngram之后的ngram_options,对ngram_options进行TFIDF统计
        :return: optionsInfo：dict
        """
        options_num = len(self.ngram_options)
        # 遍历该问题对应的所有选项
        for index in range(options_num):
            # 计算得到每个选项对应的TF值
            word_frequency = self.getTFCount(self.ngram_options[index])
            self.optionsInfo[index] = {}
            self.optionsInfo[index]['wf'] = word_frequency
        # 一个单词在多少选项中出现
        word_option_frequency = {}
        for index in range(options_num):
            for word in self.optionsInfo[index]['wf'].keys():
                if word in word_option_frequency.keys():
                    word_option_frequency[word] += 1
                else:
                    word_option_frequency[word] = 1
        # 计算每个单词的idf值
        for word in word_option_frequency.keys():
            self.idf[word] = math.log(options_num/word_option_frequency[word])
        # 计算TF*IDF
        for index in range(options_num):
            self.optionsInfo[index]["tfIdf"] = {}
            for word in self.optionsInfo[index]['wf'].keys():
                self.optionsInfo[index]["tfIdf"][word] = self.optionsInfo[index]['wf'][word] * self.idf[word]
    # def vector_distance(self, vector):
    #     """
    #     compute vector distance
    #     :param vector: dict, Term frequency. {'word': frequency}
    #     :return: distance: |vector|
    #     """
    #     distance = 0
    #     for word in vector.keys():
    #         distance += math.pow(vector[word] * self.idf[word], 2)
    #     # 公式中|x|, 需要开平方取模
    #     distance = math.pow(distance, 0.5)
    #     return distance

    def compute_similarity(self, p_info, query_vector, query_distance):
        """
        compute cosine-similarity between query and answer candidate
        :param p_info: dict, optionsInfo[index], including "wf" and "tfIdf". 一个候选答案的相关信息
        :param query_vector: dict, tf 问题的tf向量, {'单词': '频率'}
        :param query_distance: distance of query_vector
        :return: similarity: x*y/(|x| * |y|)
        """
        p_vector_distance = 0
        for word in p_info["wf"].keys():
            p_vector_distance += math.pow(p_info["tfIdf"][word], 2)
        # 公式中|x|, 需要开平方取模
        p_vector_distance = math.pow(p_vector_distance, 0.5)
        if p_vector_distance == 0:
            return 0

        # 计算x,y的点积
        dot_product = 0
        for word in query_vector.keys():
            if word in p_info["wf"].keys():
                idf = self.idf[word]
                dot_product += p_info["tfIdf"][word] * query_vector[word] * idf
        similarity = dot_product/(p_vector_distance * query_distance)
        return similarity

    def get_similar_option(self, query_vector):
        """
        get top3 similar answer candidate by compute_similarity between query and answer candidate
        :param query_vector: dict, 问题的tf向量, {'单词': '频率'}
        :return: top_options: 对options按照cosine similarity 进行排序
        """
        query_distance = 0
        for word in query_vector.keys():
            if word in self.idf.keys():
                query_distance += math.pow(query_vector[word] * self.idf[word], 2)
        query_distance = math.pow(query_distance, 0.5)
        if query_distance == 0:
            return [None]
        top_options = []
        # 如果有选项similarity == 0, 是否就不用返回top3了？
        for index in range(0, len(self.ngram_options)):
            similarity = self.compute_similarity(self.optionsInfo[index], query_vector, query_distance)
            top_options.append((index, similarity))
        return sorted(top_options, key=lambda x: x[1], reverse=True)[:3]

    def get_ngram(self, sent, ngram):
        """
        输入已经分完词的option, 去除停止词后，输出该option的nGram形式
        :param sent: ['北京科技大学', '，', '简称', '北科', '或',]
        :param ngram: ['北京科技大学','北京科技大学简称','北京科技大学简称北科']
        :return:
        """
        sent = [word for word in sent if word not in self.stopwords]
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

    def query(self, query_vector, answer_types):
        """
        input a question vector, return top3 relevant answers
        :param query_vector: dict, {word: frequency}
        :param answer_types: list, e.g. ["人物","地点"]
        :return: possible_answers: list of object,  [{"answer": 76个本科专业", "score":0.1555555}]
        """
        answers = self.get_similar_option(query_vector)
        possible_answers = []
        if answers == [None]:
            return possible_answers
        for index, sim in answers:
            # possible_answers.append(("".join(self.options[index]), sim))
            answer_str = "".join(self.options[index])
            named_entities = self.get_named_entity(answer_str)
            concrete_answers = self.get_concrete_by_answer_type(named_entities, answer_types, answer_str)
            if concrete_answers:
                print({"answer": answer_str, "score": sim, "concrete_answer": ", ".join(concrete_answers)})
            possible_answers.append({"answer": answer_str, "score": sim, "concrete_answer": ", ".join(concrete_answers)})
        return possible_answers


if __name__ == "__main__":
    sourcefile = './data/output/fileContent.json'
    with open(sourcefile, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
    for question_str in content.keys():
        question = ProcessQuestion(question_str, "./data/stopwords.txt", ngram=1)
        tfidf = RetrievalTFIDF(content[question_str]["options"], ngram=1)
        possible_answers = tfidf.query(question.question_vector, question.answer_types)
    # tfidf = RetrievalTFIDF(content["范德堡大学是由谁捐款创办的？"]["options"], ngram=1)
    # question = ProcessQuestion("范德堡大学是由谁捐款创办的？", "./data/stopwords.txt", ngram=1)
    # possible_answers = tfidf.query(question.question_vector, question.answer_type)
    # print(possible_answers)



