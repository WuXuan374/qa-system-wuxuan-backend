#!flask/bin/python
from flask import Flask, request, abort, jsonify, make_response
from flask_cors import CORS
import json
import os
import sys
sys.path.append("..")
from run_QA import ReadDocumentContent
from preprocess import get_keyword
from question_retrieval import QuestionRetrieval

app = Flask(__name__)
CORS(app)

# read file contentcontent
sourcefile = './data/TrecQA_train.json'
if os.path.isfile(sourcefile):
    with open(sourcefile, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
else:
    raise Exception("source file not exists\n")

# api configuration
apis = {
    "answers": "/api/answers/",
    "hints": "/api/hints",
}


def get_titles_by_keywords(keywords):
    """
    用户输入问题A --> 提取问题A中的关键词 --> 通过关键词查找匹配方式，寻找语料库中相似的问题B --> 返回B的title
    语料库中相似的问题可能不止一条，所以可以返回多个相似问题
    :param keywords: list
    :return: question: list, ["南京大学化学化工学院改过哪些名字？", "南京大学小百合BBS是哪年创立？", ]
    """
    questions = []
    if not keywords:
        return questions
    for question in content.keys():
        for keywordInfo in (content[question]["keywords"]):
            for query_keyword in keywords:
                # 关键词完全匹配 or 用户关键词是文档关键词的子集
                # e.g. 用户关键词：“北京大学”  文档关键词： “北京大学药学院”
                if query_keyword == keywordInfo \
                        or keywordInfo.find(query_keyword) != -1:
                    # 通过上述关键词匹配，获得文本库中所有相关文本的标题(question)
                    questions.append(question)
                    # 有一个关键词匹配，就可以跳过当前循环，去查找下一个文本
                    break
    return questions


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
        reader = ReadDocumentContent(sourcefile, ngram=1)
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
def get_keywords():
    """
    将语料库中的问题返回给前端，作为用户输入问题时的提示信息
    """
    hints = list(content.keys())
    return jsonify({'hints': hints})


if __name__ == '__main__':
    app.run(debug=True)
