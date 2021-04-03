#!flask/bin/python
from flask import Flask, request, abort, jsonify, make_response
import torch
from flask_cors import CORS
import json
import os
import sys
import pickle
sys.path.append("..")
from run_QA import ReadDocumentContent
from preprocess import get_keyword
from question_retrieval import QuestionRetrieval
from BiDAF.model import BiDAF_model
from helper import get_args, run_with_model


# app 配置及跨域
app = Flask(__name__)
CORS(app)

# 文件路径
sourceFilePath = './data/TrecQA_train.json'
checkPointPath = './data/epoch_4.pt'
vectorPath = './data/pretrained_vectors.pickle'
charVocabPath = './data/char_vocab.pickle'
wordVocabPath = './data/word_vocab.pickle'

# 基础文件读取
if os.path.isfile(sourceFilePath):
    with open(sourceFilePath, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
else:
    raise Exception("source file not exists\n")

checkPoint = torch.load(checkPointPath)
with open(vectorPath, 'rb') as handle:
    pretrained_vectors = pickle.load(handle)
with open(charVocabPath, 'rb') as handle:
    char_vocab = pickle.load(handle)
with open(wordVocabPath, 'rb') as handle:
    word_vocab = pickle.load(handle)

# 928 == len(char_vocab)
args = get_args(928)
model = BiDAF_model(args, pretrained_vectors)
model.load_state_dict(checkPoint['model_state_dict'])
# run_with_model(model, questions, contexts, word_vocab, char_vocab, lang="en")

# api configuration
apis = {
    "answers": "/api/answers/",
    "hints": "/api/hints",
}


@app.route(apis["answers"], methods=['GET'])
def get_answers():
    """
    用户输入问题--> 返回top3候选答案
    用户输入问题A --> 通过计算tfIdf cosine similarity,寻找语料库中相似的问题B --> 在问题B对应的文本中查找候选答案
    :return: 404： 用户输入的问题为空/没有在语料库中找到和这个问题相关的文本
    :return: 200： {'answers': answers}
    """
    question_str = request.args.get("question")
    stop_word_path = './data/stopwords.txt'
    if question_str is None or len(question_str) == 0:
        return make_response(jsonify({'error': 'Not found'}), 404)
    else:
        reader = ReadDocumentContent(sourceFilePath, ngram=1)
        question_options = list(content.keys())
        question_retrieval = QuestionRetrieval(question_str, question_options, top_num=3)
        # question_titles: list, e.g ['《野猪历险记》的游戏语言是什么？', ]
        question_titles = list(map(lambda option: option[0], question_retrieval.candidate_options))
        # sorted_answers: [answer, score, document_title]
        sorted_answers = []
        if not question_titles:
            return make_response(jsonify({'error': 'Not found'}), 404)
        # 根据title, 在title对应的文本中查找答案
        for title in question_titles:
            current_answer = reader.get_question_answer\
                (question_str, content[title]["options"], stop_word_path)
            for item in current_answer:
                item["document_title"] = title
                sorted_answers.append(item)
        # 从多个文本中，每个文本收集三个答案，随后对收集到的所有答案再根据score进行排序
        sorted_answers = sorted(sorted_answers, key=lambda x: x["score"], reverse=True)[:3]
        return jsonify({'answers': sorted_answers})


@app.route(apis["hints"], methods=['GET'])
def get_hints():
    """
    将语料库中的问题返回给前端，作为用户输入问题时的提示信息
    """
    hints = list(content.keys())
    return jsonify({'hints': hints})


if __name__ == '__main__':
    app.run(debug=True)
