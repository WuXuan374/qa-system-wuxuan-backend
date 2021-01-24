from ltp import LTP
import os


class ProcessQuestion:
    def __init__(self, question, stop_word_path, remove_stopword=True):
        self.remove_stopword = remove_stopword
        if os.path.isfile(stop_word_path):
            fp = open(stop_word_path, 'r', encoding='utf-8')
            self.stopwords = [line.strip('\n') for line in fp.readlines()]
        else:
            raise Exception("stop words file not exists!\n")
        self.question = question
        self.tokenized_question = self.tokenize(question)
        self.question_vector = self.get_vector(self.tokenized_question)

    def tokenize(self, question):
        """
        结合分词技术和命名实体识别， 对问题进行分词
        :param question: str, 未分词："兼善中学是重庆市的一所中学，位于北碚区毛背沱。"
        :return: sent: list, 分词之后的数组: ['兼善中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区毛背沱', '。']
        """
        ltp = LTP()
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
        return sent

    def get_vector(self, tokenized_question):
        """
        获得问题对应的TF向量
        :param tokenized_question: 已经分过词的问题 ['兼善中学', '是', '重庆市', '的', '一', '所', '中学', '，', '位于', '北碚区毛背沱',
        :return: dict, {word: frequency}
        """
        word_frequency = {}
        for word in tokenized_question:
            if self.remove_stopword and word in self.stopwords:
                continue
            if word in word_frequency.keys():
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        return word_frequency
