#!flask/bin/python
from flask import Flask, request, abort, jsonify, make_response
from flask_cors import CORS
import json
import os
import sys
sys.path.append("..")
from run_QA import ReadDocumentContent
from preprocess import get_keyword

app = Flask(__name__)
CORS(app)

# read file content
sourcefile = './data/fileContent.json'
if os.path.isfile(sourcefile):
    with open(sourcefile, 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
else:
    raise Exception("source file not exists\n")

# api configuration
apis = {
    "answers": "/api/answers/",
    "keywords": "/api/keywords",
}


def get_title_by_keywords(keywords):
    """
    用户输入问题A --> 提取问题A中的关键词 --> 通过关键词查找匹配方式，寻找语料库中相似的问题B --> 返回B的title
    :param keywords: list
    :return: question: string
    """
    if not keywords:
        return None
    for question in content.keys():
        for keywordInfo in (content[question]["keywords"]):
            if keywords[0] == keywordInfo[1]:
                return question
    return None


@app.route(apis["answers"], methods=['GET'])
def get_answers():
    """
    用户输入问题--> 返回top3候选答案
    用户输入问题A --> 提取问题A中的关键词 -->
    通过关键词查找匹配方式，寻找语料库中相似的问题B --> 在问题B对应的文本中查找候选答案
    :return: 404： 用户输入的问题为空/没有在语料库中找到和这个问题相关的文本
    :return: 200： {'answers': answers}
    """
    question_str = request.args.get("question")
    stop_word_path = './data/stopwords.txt'
    if question_str is None or len(question_str) == 0:
        make_response(jsonify({'error': 'Not found'}), 404)
    else:
        reader = ReadDocumentContent(sourcefile)
        keywords = get_keyword(question_str)
        question_title = get_title_by_keywords(keywords)
        if question_title is None:
            return make_response(jsonify({'error': 'Not found'}), 404)
        answers = reader.get_question_answer(
            question_str, content[question_title]["options"], stop_word_path)
        return jsonify({'answers': answers})


@app.route(apis["keywords"], methods=['GET'])
def get_keywords():
    """
    从语料库中提取关键词，返回给前端。用户可以根据关键词提示来进行提问
    """
    keywords = []
    for question in content.keys():
        for tag, keyword in content[question]["keywords"]:
            keywords.append((tag, keyword))
    return jsonify({'keywords': keywords})


if __name__ == '__main__':
    app.run(debug=True)
