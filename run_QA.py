from retrieval_TFIDF import RetrievalTFIDF
from process_question import ProcessQuestion
import json


# 创建这个类的目的：提前完成json文件的读取，不需要针对每个问题去读取一次文件
class ReadDocumentContent:
    def __init__(self, source_file, ngram=1):
        self.source_file = source_file
        with open(source_file, 'r', encoding="utf-8") as load_j:
            self.content = json.load(load_j)
        self.ngram = ngram

    def get_question_answer(self, question_str, answer_options, stop_word_path):
        """
        input a question string, and get top possible answers
        :param question_str: string, e.g. "重庆大学建筑学部坐落在哪里？"
        :param answer_options: list of list, e.g. [ [ "2010年","，","世宗大学",]]
        :param stop_word_path: path, e.g. "./data/stopwords.txt"
        :return: possible_answers: list of tuple, [(answer(str), similarity(number)] [("76个本科专业", 0.1555555)]
        """
        question = ProcessQuestion(question_str, stop_word_path, ngram=self.ngram)
        tfidf = RetrievalTFIDF(answer_options, question.answer_types, ngram=self.ngram)
        possible_answers = tfidf.query(question.question_vector)
        return possible_answers


if __name__ == "__main__":
    sourcefile = './data/output/fileContent.json'
    reader = ReadDocumentContent(sourcefile, ngram=2)
    answers = reader.get_question_answer("大分短期大学在什么地方？",
                                         reader.content["大分短期大学在什么地方？"]["options"],
                                         stop_word_path="./data/stopwords.txt")
    print(answers)
