import json
import os
from run_QA import ReadDocumentContent
from process_question import ProcessQuestion


class Evaluation:
    def __init__(self, source_file):
        self.source_file = source_file
        if not os.path.isfile(source_file):
            raise Exception("Source file not exists")
        self.reader = ReadDocumentContent(self.source_file)
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
    evaluation = Evaluation("./data/output/fileContent.json")
    mrr, accuracy = evaluation.compute_metrics()
    print("*******First version evaluation")
    print("MRR: ", mrr)
    print("accuracy: ", accuracy)
    result = {
        "first_version": {
            "MRR": mrr,
            "accuracy:": accuracy,
        }
    }
    with open('./data/output/evaluation.json', 'a', encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
