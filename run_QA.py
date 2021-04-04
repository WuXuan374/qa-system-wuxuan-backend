from retrieval_TFIDF import RetrievalTFIDF
from process_question import ProcessQuestion
import json
from reorder import Reorder
from logistic_regression import Logistic_Regression
import torch
import os


# 创建这个类的目的：提前完成json文件的读取，不需要针对每个问题去读取一次文件
class ReadDocumentContent:
    def __init__(self, source_file, ngram=1):
        self.source_file = source_file
        with open(source_file, 'r', encoding="utf-8") as load_j:
            self.content = json.load(load_j)
        self.ngram = ngram

    def get_question_answer(self, question_str, answer_options, stop_word_path, lang="zh"):
        """
        input a question string, and get top possible answers
        :param question_str: string, e.g. "重庆大学建筑学部坐落在哪里？"
        :param answer_options: list of list, e.g. [ [ "2010年","，","世宗大学",]]
        :param stop_word_path: path, e.g. "./data/stopwords.txt"
        :param lang: "zh" | "en"
        :return: possible_answers: list of tuple, [(answer(str), similarity(number)] [("76个本科专业", 0.1555555)]
        """
        # question = ProcessQuestion(question_str, stop_word_path, self.embeddings, ngram=self.ngram)
        question = ProcessQuestion(question_str, stop_word_path, ngram=self.ngram)
        # word_embedding = RetrievalWordEmbedding(answer_options, question.answer_types, self.embeddings, ngram=self.ngram)
        # possible_answers = word_embedding.query(question.question_embedding)
        tfidf = RetrievalTFIDF(answer_options, question.answer_types, ngram=self.ngram, lang=lang)
        possible_answers = tfidf.query(question.question_vector)
        # 计算重排序得分
        reorder = Reorder(lang=lang)
        for possible_answer in possible_answers:
            m_score = reorder.compute_score(question_str, possible_answer["answer"], (1, 3))
            possible_answer["second_score"] = m_score
            # 将重排序得分和初次检索得分进行相加
            model = Logistic_Regression(2, 2)
            model.load_state_dict(torch.load('F:/QA-system-wuxuan/qa-system-wuxuan/Logistic_Regression/reorder_param.pkl'))
            score_tensor = torch.tensor([possible_answer["first_score"], m_score], dtype=torch.float).view(-1, 2)
            final_score, _ = model(score_tensor)
            final_score = final_score[0][1]
            possible_answer["final_score"] = final_score.item()
        # 根据相加后的分数，再次排序
        possible_answers = sorted(possible_answers, key=lambda x: x["final_score"], reverse=True)
        return possible_answers


if __name__ == "__main__":
    sourcefile = './data/output/fileContent.json'
    reader = ReadDocumentContent(sourcefile, ngram=1)
    answers = reader.get_question_answer("大分短期大学在什么地方？",
                                         reader.content["大分短期大学在什么地方？"]["options"],
                                         stop_word_path="./data/stopwords.txt")
    print(answers)
    # reorder = Reorder()

