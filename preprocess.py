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

ltp = LTP()


def file_content_to_dict(f):
    return {
        'keywords': f.keywords,
        'right answer': f.right_answer,
        'options': f.answer_options,
    }


def to_string(array):
    """
    :param array: ['复旦', '大学']
    :return: '复旦大学'
    """
    return ''.join(array)


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
            print(question)
            if res.get(question) is None:
                res[question] = FileContent()
            res[question].add_answer_options(preprocess(content))
            if flag == '1':
                res[question].set_right_answer(preprocess(content))

    else:
        raise Exception(filename + " not exists\n")

    return res


def keyword_extraction(file_content):
    """
    extract keyword from question using named-entity recognition
    :param file_content: FileContent()
    :return:
    """

    # [question, question....]
    for key, value in file_content.items():
        seg, hidden = ltp.seg([key])
        # ner: [[('Nh', 2, 2)]]
        ner = ltp.ner(hidden)
        # keywords: [('PERSON', "吴轩")],  tuple_item: ('Nh', 2, 2)
        keywords = [(tag_to_name[tuple_item[0]], to_string(seg[0][tuple_item[1]: tuple_item[2]+1])) for tuple_item in ner[0]]
        file_content[key].keywords = keywords

    return file_content


def get_keyword(question_str):
    """
    input a question string, extract its keyword using named-entity recognition
    :param question_str: str, e.g. "布里斯本商学研究所的排名如何？"
    :return: keywords: list, e.g. ["布里斯本", "商学研究所"]
    """
    seg, hidden = ltp.seg([question_str])
    # ner: [[('Nh', 2, 2)]]
    ner = ltp.ner(hidden)
    # keywords: [('PERSON', "吴轩")],  tuple_item: ('Nh', 2, 2)
    keywords = [to_string(seg[0][tuple_item[1]: tuple_item[2] + 1]) for tuple_item in
                ner[0]]
    return keywords


def preprocess(option):
    """
    结合分词技术和命名实体识别， 对候选答案进行预处理
    :param option: str, 未分词："兼善中学是重庆市的一所中学，位于北碚区毛背沱。"
    :return: sent: list, 分词之后的数组: ['兼善中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区毛背沱', '。']
    """

    seg, hidden = ltp.seg([option])
    # ['兼善', '中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区', '毛背沱', '。']
    sent = seg[0]
    # ner[0]: [('Ni', 0, 1), ('Ns', 3, 3), ('Ns', 10, 11)]
    ner = ltp.ner(hidden)
    for i in range(len(ner[0])-1, -1, -1):
        _, start, end = ner[0][i]
        # '北碚区毛背沱' 代替 '北碚区', '毛背沱'
        sent[start] = ''.join(sent[start:end+1])
        # 记得pop时也要倒序pop
        for j in range(end, start, -1):
            sent.pop(j)
    return sent


if __name__ == "__main__":
    # res = read_tsv_file('./data/ChineseDBQA/nlpcc2017.dbqa.train')
    # file_content = keyword_extraction(res)
    # with open('./data/output/fileContent.json', 'w', encoding="utf-8") as fp:
    #     json.dump(file_content, fp, indent=4, ensure_ascii=False, default=file_content_to_dict)
    keywords = get_keyword("新北市私立辞修高级中学学校名称的由来是什么？")
    print(keywords)



