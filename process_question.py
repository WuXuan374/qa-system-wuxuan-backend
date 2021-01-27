from ltp import LTP
import os
import json
ltp = LTP()


class ProcessQuestion:
    def __init__(self, question, stop_word_path, ngram=1):
        if os.path.isfile(stop_word_path):
            fp = open(stop_word_path, 'r', encoding='utf-8')
            self.stopwords = [line.strip('\n') for line in fp.readlines()]
        else:
            raise Exception("stop words file not exists!\n")
        self.question = question
        # self.hidden是LTP产生的一个数组，后面的词性标注会用到
        self.tokenized_question, self.hidden = self.tokenize(question)
        # 去除停止词，生成ngram. 参数为1时，直接采用tokenized_question
        self.ngram_question = self.get_ngram(self.tokenized_question, ngram=ngram)
        # self.answer_type = self.determine_answer_type()
        self.question_vector = self.get_vector(self.ngram_question)
        self.answer_type = self.determine_answer_type()


    def tokenize(self, question):
        """
        结合分词技术和命名实体识别， 对问题进行分词
        :param question: str, 未分词："兼善中学是重庆市的一所中学，位于北碚区毛背沱。"
        :return: sent: list, 分词之后的数组: ['兼善中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区毛背沱', '。']
        :return: hidden: LTP分词所产生的的数组，需要用于后面的词性标注任务
        """
        seg, hidden = ltp.seg([question])
        # ['兼善', '中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区', '毛背沱', '。']
        sent = seg[0]
        # ner[0]: [('Ni', 0, 1), ('Ns', 3, 3), ('Ns', 10, 11)]
        ner = ltp.ner(hidden)
        for i in range(len(ner[0]) - 1, -1, -1):
            _, start, end = ner[0][i]
            # '北碚区毛背沱' 代替 '北碚区', '毛背沱'
            sent[start] = ''.join(sent[start:end + 1])
            # 记得pop时也要倒序pop
            for j in range(end, start, -1):
                sent.pop(j)
        return sent, hidden

    def get_vector(self, tokenized_question):
        """
        获得问题对应的TF向量
        :param tokenized_question: 已经分过词的问题 ['兼善中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区毛背沱',
        :return: dict, {word: frequency}
        """
        word_frequency = {}
        for word in tokenized_question:
            if word in self.stopwords:
                continue
            if word in word_frequency.keys():
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        return word_frequency

    def get_ngram(self, tokenized_question, ngram):
        """
        输入已经分完词的question, 去除停止词后，输出该question的nGram形式
        :param tokenized_question: ['北京科技大学', '，', '简称', '北科', '或',]
        :param ngram: num
        :return: ngram_list: ['北京科技大学','北京科技大学简称','北京科技大学简称北科']
        """
        tokenized_question = [word for word in tokenized_question if word not in self.stopwords]
        if ngram == 1:
            return tokenized_question
        question_len = len(tokenized_question)
        if question_len < ngram:
            return tokenized_question
        ngram_list = ["".join(tokenized_question[index: index+k])
                      for k in range(1, ngram + 1) for index in range(0, question_len - ngram + 1)]
        return ngram_list

    def determine_answer_type(self):
        """
        根据一些人为设定的规则，判断问句所对应的回答类型
        :return: answer_type: list, e.g. ["人物", "地点”]
        """
        answer_type = []
        for token in self.ngram_question:
            if "谁" in token or "何人" in token:
                answer_type.append("人物")
            elif "哪里" in token or "何处" in token or "什么地方" in token or "在哪儿" in token or "在哪里" in token:
                answer_type.append("地点")
            elif "何时" in token or "什么时候" in token:
                answer_type.append("日期")
            elif "多少" in token:
                answer_type.append("数量")
            elif "多久" in token or "多长" in token or "多远" in token:
                answer_type.append("线性度量")
        # 去重
        answer_type = list(set(answer_type))
        return answer_type


if __name__ == "__main__":
    with open("./data/output/fileContent.json", 'r', encoding="utf-8") as load_j:
        content = json.load(load_j)
    for question_str in content.keys():
        question = ProcessQuestion(question_str, './data/stopwords.txt', ngram=2)
        question.determine_answer_type()
