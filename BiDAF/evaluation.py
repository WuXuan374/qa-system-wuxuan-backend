import json
import argparse
import nltk
from nltk.corpus import stopwords
import string


def preprocess(tokens):
    """
    对预测答案和实际答案进行预处理，包括去除停止词、标点符号等
    """
    stopwords_list = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords_list]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token.lower() for token in tokens]
    return tokens


def exact_match(prediction, answer):
    return ' '.join(preprocess(prediction.split())) == ' '.join(preprocess(answer.split()))


def f1_score(prediction, answer):
    prediction_tokens = preprocess(prediction.split())
    answer_tokens = preprocess(answer.split())
    same_len = len([token for token in prediction_tokens if token in answer_tokens])
    if same_len == 0:
        return 0
    p_score = 1.0 * same_len/len(answer_tokens)
    r_score = 1.0 * same_len/len(prediction_tokens)
    f1 = 2 * p_score * r_score / (p_score + r_score)
    return f1


def best_metric_over_candidate_answers(fn, prediction, candidate_answers):
    """
    在Squad dev数据集中, 针对每个问题给出了多个正确答案，从而提升数据集的鲁棒性。
    我们将预测的答案和候选答案进行一一对比，选取最好的结果
    :param fn: 计算相应metric的函数
    :param prediction: 预测答案, string
    :param candidate_answers: list of string
    :return:
    """
    metrics = []
    for answer in candidate_answers:
        metric = fn(prediction, answer)
        metrics.append(metric)
    return max(metrics)


def evaluate(dataset, predictions):
    """
    :param dataset: see dev-v1.1.json
    :param predictions: dict, e.g. {"id": answer(str)}
    :return:
    """
    F1, EM, total_num = 0, 0, 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                id = qas["id"]
                answers = qas["answers"]
                if id not in predictions:
                    print('unanswered question: {}, score will be recorded as 0'.format(id))
                    continue
                candidate_answers = list(map(lambda x: x["text"], answers))
                prediction = predictions[id]
                F1 += best_metric_over_candidate_answers(f1_score, prediction, candidate_answers)
                EM += best_metric_over_candidate_answers(exact_match, prediction, candidate_answers)
                total_num += 1
    F1 = 100.0 * F1 / total_num
    EM = 100.0 * EM / total_num
    return {'EM': EM, 'F1': F1}


def main(args):
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    results = evaluate(dataset, predictions)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', )
    parser.add_argument('--prediction_file',)
    args = parser.parse_args()
    main(args)
