import json
import os
from run_QA import ReadDocumentContent
from gensim.models import KeyedVectors


class Evaluation:
    def __init__(self, source_file, ngram=1, word_vector=None):
        self.source_file = source_file
        if not os.path.isfile(source_file):
            raise Exception("Source file not exists")
        self.ngram = ngram
        self.reader = ReadDocumentContent(self.source_file, self.ngram, word_vector=word_vector)
        self.content = self.reader.content

    def compute_metrics(self, lang="zh", type="TFIDF"):
        """
        compute evaluation metrics: MRR and Accuracy
        :param type: "TFIDF" || "embedding", 计算TFIDF矩阵的相似度 || 计算 Word Embedding 的相似度
        :return: mrr(num)
        :return: accuracy(num)
        """
        question_num = len(self.content.keys())
        print("question num:", question_num)
        mrr_sum = 0
        accuracy_sum = 0
        for question_str in self.content.keys():
            right_answer = self.content[question_str]["right answer"]
            if right_answer == "":
                mrr_sum += 1
                accuracy_sum += 1
                continue
            predicted_answers = self.reader.get_question_answer(
                question_str, self.content[question_str]["options"], stop_word_path="./data/stopwords.txt", lang=lang, type=type)
            if not predicted_answers:
                continue
            for index in range(len(predicted_answers)):
                if (" ".join(right_answer)) == (predicted_answers[index]["answer"]):
                    mrr_sum += 1/(index+1)
                    accuracy_sum += 1
                    print("question", question_str)
                    print("right answer" + " ".join(right_answer).strip())
                    print(predicted_answers[index])
                    break
        mrr = (1/question_num) * mrr_sum
        accuracy = (1/question_num) * accuracy_sum
        return mrr, accuracy


if __name__ == "__main__":
    word_vector = KeyedVectors.load('./data/word2vec/vectors.kv')
    evaluation = Evaluation("./data/TFIDF_input/TrecQA_train.json", ngram=1, word_vector=word_vector)
    mrr, accuracy = evaluation.compute_metrics(lang="en", type="embedding")
    print("*******12th version TrecQA_train(embedding)******")
    print("MRR: ", mrr)
    print("accuracy: ", accuracy)
    result = {
        "12th version TrecQA_train(embedding)": {
            "MRR": mrr,
            "accuracy:": accuracy,
        }
    }
    with open('./data/output/evaluation.json', 'a', encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
