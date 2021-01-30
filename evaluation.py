import json
import os
from run_QA import ReadDocumentContent


class Evaluation:
    def __init__(self, source_file, ngram=1):
        self.source_file = source_file
        if not os.path.isfile(source_file):
            raise Exception("Source file not exists")
        self.ngram = ngram
        self.reader = ReadDocumentContent(self.source_file, self.ngram)
        self.content = self.reader.content

    def compute_metrics(self):
        """
        compute evaluation metrics: MRR and Accuracy
        :return: mrr(num)
        :return: accuracy(num)
        """
        question_num = len(self.content.keys())
        print("question num:", question_num)
        mrr_sum = 0
        accuracy_sum = 0
        for question_str in self.content.keys():
            right_answer = self.content[question_str]["right answer"]
            predicted_answers = self.reader.get_question_answer(
                question_str, self.content[question_str]["options"], stop_word_path="./data/stopwords.txt")
            if not predicted_answers:
                continue
            for index in range(len(predicted_answers)):
                if ("".join(right_answer)) == (predicted_answers[index]["answer"]):
                    mrr_sum += 1/(index+1)
                    accuracy_sum += 1
                    print("question", question_str)
                    print("right answer" + "".join(right_answer).strip())
                    print(predicted_answers[index])
                    break
        mrr = (1/question_num) * mrr_sum
        accuracy = (1/question_num) * accuracy_sum
        return mrr, accuracy


if __name__ == "__main__":
    evaluation = Evaluation("./data/output/fileContent.json", ngram=2)
    mrr, accuracy = evaluation.compute_metrics()
    print("*******5th version(word2vec) evaluation(ngram=2)")
    print("MRR: ", mrr)
    print("accuracy: ", accuracy)
    result = {
        "5th version(word2vec)(ngram=2)": {
            "MRR": mrr,
            "accuracy:": accuracy,
        }
    }
    with open('./data/output/evaluation.json', 'a', encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
