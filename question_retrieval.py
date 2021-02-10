import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance


class QuestionRetrieval:
    def __init__(self, question_str, options, top_num):
        self.idf = {}  # dict
        self.question_str = self.tokenize(question_str)
        self.options = options
        self.tokenized_options = list(map(lambda option: self.tokenize(option), options))
        self.tfIdf_array = self.compute_tfidf()
        self.candidate_options = self.compute_similarity(top_num=top_num)

    def compute_tfidf(self):
        """
        将用户输入的问题和语料库中的问题合起来作为训练材料，计算得出tfidf矩阵
        该矩阵的第一行是用户输入的问题
        :return: result_array (5678, 13913)
        """
        corpus = [self.question_str] + self.tokenized_options
        # 参数说明: token_pattern: 该正则表达式保证会把单个字构成的单词纳入考虑（默认不会）
        # max_df = 0.4: 在超过40%文章中出现的单词会被舍弃
        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.4)
        result = tfidf_model.fit_transform(corpus)
        result_array = result.toarray()
        return result_array

    def compute_similarity(self, top_num=3):
        """
        通过计算用户输入的问题和语料库中问题的cosine similarity， 找出最为相似的问题
        :param top_num: 返回的候选答案数量
        :return: candidate_options: list, e.g. [('《野猪历险记》的游戏语言是什么？', 1.0)]
        """
        candidate_options = []
        for index in range(len(self.options)):
            # cosine similarity = 1- cosine distance
            sim = 1 - distance.cosine(self.tfIdf_array[0], self.tfIdf_array[index+1])
            if sim > 0.0:
                candidate_options.append((self.options[index], sim))
        candidate_options = sorted(candidate_options, key=lambda x: x[1], reverse=True)[:top_num]
        return candidate_options

    def tokenize(self, str):
        """
        对句子进行分词。包含去除停止词功能，支持生成n-gram
        :param str: string
        :param ngram: if ngram=2, generates 1-gram and 2-gram
        :return: sent: string, 将分词后的结果通过空格分隔(Tfidf Vectorizer的要求）
                e.g. '今天 是 晴天'
        """
        # 分词：精确模式
        word_list = jieba.cut(str, cut_all=False)
        # 去除停止词
        # word_list = [word for word in word_list if word not in self.stopwords and not word.isspace()]
        word_list = [word for word in word_list if not word.isspace()]
        sent = " ".join([word for word in word_list])
        return sent


if __name__ == "__main__":
    sourcefile = './data/TFIDF_input/train_2016_new.json'
    with open(sourcefile, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
    # for question_str in content.keys():
    #     question = ProcessQuestion(question_str, "./data/stopwords.txt", ngram=1)
    #     tfidf = RetrievalTFIDF(content[question_str]["options"], question.answer_types, ngram=1)
    #     possible_answers = tfidf.query(question.question_vector, question.answer_types)
    # question = ProcessQuestion("重庆市兼善中学是谁、在哪一年建立的？", "./data/stopwords.txt", ngram=2)
    # tfidf = RetrievalTFIDF(content["重庆市兼善中学是谁、在哪一年建立的？"]["options"],
    #                        question.answer_types, ngram=2)
    # possible_answers = tfidf.query(question.question_vector)
    # print(possible_answers)
    question_options = list(content.keys())
    question_retrieval = QuestionRetrieval("《野猪历险记》的游戏语言是什么？", question_options, top_num=3)
    print(question_retrieval.candidate_options)


