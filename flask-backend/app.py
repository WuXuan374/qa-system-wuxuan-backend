#!flask/bin/python
from flask import Flask, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

answers = {
    '吴轩': '电子科技大学',
    '郭谢翔': '福州大学',
    '余海杰': '福建华侨大学',
}


@app.route('/api/answers/', methods=['GET'])
def get_answers():
    question = request.args.get("question")
    if question is None or len(question) == 0:
        abort(404)
    else:
        return answers[question]


if __name__ == '__main__':
    app.run(debug=True)
