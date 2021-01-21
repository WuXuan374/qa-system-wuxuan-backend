import os
import csv
from ltp import LTP
import json


class FileContent:
    def __init__(self):
        self.answer_options = []
        self.right_answer = ""
        self.keywords = []

    def add_answer_options(self, option):
        self.answer_options.append(option)

    def set_right_answer(self, answer):
        self.right_answer = answer


tag_to_name = {
    'Nh': '人名',
    'Ni': '机构名',
    'Ns': '地名',
}


def file_content_to_dict(f):
    return {
        'keywords': f.keywords,
        'right answer': f.right_answer,
        'options': f.answer_options,
    }


def read_tsv_file(filename):
    """
    :param filename: string
    :return: res: dictionary
    """
    res = dict()
    if os.path.isfile(filename):
        tsv_file = open(filename, 'r', encoding="utf-8")
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            flag, question, content = row
            if res.get(question) is None:
                res[question] = FileContent()
            res[question].add_answer_options(content)
            if flag == '1':
                res[question].set_right_answer(content)

    else:
        raise Exception(filename + " not exists\n")

    return res


def keyword_extraction(file_content):
    """
    extract keyword from question using named-entity recognition
    :param file_content: FileContent()
    :return:
    """
    ltp = LTP()

    def to_string(array):
        """
        :param array: ['复旦', '大学']
        :return: '复旦大学'
        """
        return ''.join(array)

    # [question, question....]
    for key, value in file_content.items():
        seg, hidden = ltp.seg([key])
        # ner: [[('Nh', 2, 2)]]
        ner = ltp.ner(hidden)
        # keywords: [('PERSON', "吴轩")],  tuple_item: ('Nh', 2, 2)
        keywords = [(tag_to_name[tuple_item[0]], to_string(seg[0][tuple_item[1]: tuple_item[2]+1])) for tuple_item in ner[0]]
        file_content[key].keywords = keywords
    return file_content


if __name__ == "__main__":
    res = read_tsv_file('./data/ChineseDBQA/nlpcc2017.dbqa.train')
    file_content = keyword_extraction(res)
    with open('./data/output/fileContent.json', 'w', encoding="utf-8") as fp:
        json.dump(file_content, fp, indent=4, ensure_ascii=False, default=file_content_to_dict)



