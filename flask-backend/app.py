#!flask/bin/python
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
import json
import os
import sys
sys.path.append("..")
from run_QA import ReadDocumentContent

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


@app.route(apis["answers"], methods=['GET'])
def get_answers():
    question_str = request.args.get("question")
    stop_word_path = './data/stopwords.txt'
    if question_str is None or len(question_str) == 0:
        abort(404)
    else:
        reader = ReadDocumentContent(sourcefile)
        answers = reader.get_question_answer(question_str, stop_word_path)
        return jsonify({'answers': answers})


@app.route(apis["keywords"], methods=['GET'])
def get_keywords():
    keywords = []
    for question in content.keys():
        for tag, keyword in content[question]["keywords"]:
            keywords.append((tag, keyword))
    return jsonify({'keywords': keywords})


if __name__ == '__main__':
    app.run(debug=True)
