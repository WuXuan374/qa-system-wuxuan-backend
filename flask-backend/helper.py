import argparse
import torch
from torch import nn
import nltk


def get_args(char_vocab_len):
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--char_dim', default=8, type=int)
    parser.add_argument('--char_channel_size', default=100, type=int)
    parser.add_argument('--char_channel_width', default=5, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--learning_rate', default=0.5, type=int)
    parser.add_argument('--char_vocab_size', default=char_vocab_len, type=int)
    
    args = parser.parse_args()
    return args


def get_titles_by_keywords(keywords, content):
    """
    用户输入问题A --> 提取问题A中的关键词 --> 通过关键词查找匹配方式，寻找语料库中相似的问题B --> 返回B的title
    语料库中相似的问题可能不止一条，所以可以返回多个相似问题
    :param keywords: list
    :param content: object: 读取语料(如 TrecQa_train.json) 得到的结果
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


# 模型输入的格式
class Testcase:
    def __init__(self, c_char, q_char, c_word, q_word):
        self.c_char = c_char
        self.q_char = q_char
        self.c_word = c_word
        self.q_word = q_word


def word_tokenize(str, lang="en"):
    return [token.replace("''", '"').replace("``", '"')
            for token in nltk.word_tokenize(str)]


def run_with_model(model, questions, contexts, word_vocab, char_vocab, lang="en"):
    """
    :param model:
    :param questions: list of question(str)
    :param contexts: list of context(str)
    :param word_vocab:
    :param char_vocab:
    :param lang: "en" | "zh"
    :return: answers: list of answer(str)
    """

    with torch.no_grad():
        # Word Embedding
        question_tokens = [[word_vocab[token] for token in word_tokenize(question, lang)] for question in questions]
        context_tokens = [[word_vocab[token] for token in word_tokenize(context, lang)] for context in contexts]
        print(question_tokens)
        print(context_tokens)
        q_word = (torch.tensor(question_tokens), torch.tensor(list(map(lambda question: len(question), question_tokens))))
        print(q_word)
        # q_word[0]: [batch_size, 8], q_word[1]: [batch_size]
        c_word = (torch.tensor(context_tokens), torch.tensor(list(map(lambda context: len(context), context_tokens))))
        
        # Char Embedding
        question_char_len = max([len(token) for question in questions for token in word_tokenize(question, lang)])
        context_char_len = max([len(token) for context in contexts for token in word_tokenize(context, lang)])
        question_chars = [[[char_vocab[char] for char in token] + [0] * (question_char_len - len(token)) for token in word_tokenize(question, lang)]
                          for question in questions]
        context_chars = [[[char_vocab[char] for char in token] + [0] * (context_char_len - len(token)) for token in word_tokenize(context, lang)]
                         for context in contexts]
        # q_char: [batch_size, 8, 5]  c_char: [batch_size, 147, 11]
        q_char = torch.tensor(question_chars)
        c_char = torch.tensor(context_chars)
        
        # 通过模型 预测答案
        test_examples = Testcase(c_char, q_char, c_word, q_word)
        p_start, p_end = model(test_examples)
        softmax = nn.Softmax(dim=1)
        start_idx = torch.argmax(softmax(p_start), dim=1)
        end_idx = torch.argmax(softmax(p_end), dim=1)
        answers = []
        
        # c_char.size(0): 测试用例的数量
        for i in range(c_char.size(0)):
            answer = (word_tokenize(contexts[i], lang))[start_idx[i]: end_idx[i]+1]
            answers.append(" ".join(answer))

        return answers

