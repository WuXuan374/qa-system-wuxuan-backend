import gensim
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class CnnModel(nn.Module):
    def __init__(self, embedding_file, embedding_dim, vocab_size, filter_size=(2, 3, 4), dropout_rate=0.1):
        super(CnnModel, self).__init__()
        # 参数设置
        self.embedding_dim = embedding_dim
        self.num_of_filter = self.embedding_dim
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.embedding_file = embedding_file
        self.embedding_layer, self.word2idx = self.read_word2vec()
        # 加1是因为oov token的存在
        self.dict_size = len(self.word2idx.keys())+1
        print(self.dict_size)
        self.oov_index = self.dict_size-1

        # 模型
        # 多种conv model的集合，区别在于kernel_size不同： (2,3,4)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.num_of_filter,
                      kernel_size=(h, self.embedding_dim)) for h in self.filter_size
        ])
        self.dropout = nn.Dropout(self.dropout_rate)

    def read_word2vec(self):
        """
        read embedding model
        :return: embedding: torch embedding layer
        :return: word2index, dict, e.g.{"今天":1,}
        """
        # Load word2vec pre-train model
        embedding_model = gensim.models.KeyedVectors. \
            load_word2vec_format(self.embedding_file, binary=False, unicode_errors='ignore')
        weights_matrix = torch.FloatTensor(embedding_model.wv.vectors)
        # oov token
        weights_matrix = torch.cat((weights_matrix, torch.rand((1, self.embedding_dim), dtype=torch.float)), 0)
        # Build nn.Embedding() layer
        embedding = nn.Embedding.from_pretrained(weights_matrix)
        embedding.requires_grad = False
        word2index = {token: token_index for token_index, token in enumerate(embedding_model.index2word)}

        return embedding, word2index

    def sentence_encode(self, sentences):
        """
        将文本表示成bag-of-words格式, 不在word2idx中的词，分配一个oov_index
        :param sentences: list of list, [["今天", "天气", "如何"],]
        :return:sentences_vector： (len(sentences), vocab_size), 每一行是一个句子对应的bag-of-words向量
        """
        sentences_vector = torch.zeros([len(sentences), 1, self.vocab_size], dtype=torch.long)
        for i in range(len(sentences)):
            sentence = sentences[i]
            # oov token: 0
            vector = torch.full((1, self.vocab_size), self.oov_index, dtype=torch.long)
            for j in range(min(len(sentence), self.vocab_size)):
                word = sentence[j]
                if word in self.word2idx.keys():
                    vector[0, j] = self.word2idx[word]
            sentences_vector[i, :] = vector
        return sentences_vector

    def conv_and_pool(self, x, conv):
        """

        :param x: (1, 1, vocab_size, embedding_dim)
        :param conv:
        :return:
        """
        # 进行卷积，对卷积的结果使用Relu激活函数（使得结果范围在[0,1)之间）
        x = F.tanh(conv(x)).squeeze(3)
        # after: x (1, embedding_dim, 604/603/602)
        # size(2): out_channels
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (1, embedding_dim)
        return x

    def forward(self, question, answer):
        # out: (1, vocab_size, embedding_dim)
        q_out = self.embedding_layer(question)
        a_out = self.embedding_layer(answer)
        # out(1, 1, vocab_size, embedding_dim)
        q_out = q_out.unsqueeze(1)
        a_out = a_out.unsqueeze(1)
        # conv 需要四维的输入
        # out (1, 3*embedding_dim), 3--> 存在3种Conv2d模型
        q_out = torch.cat([self.conv_and_pool(q_out, conv) for conv in self.convs], 1)
        a_out = torch.cat([self.conv_and_pool(a_out, conv) for conv in self.convs], 1)
        q_out = self.dropout(q_out)
        a_out = self.dropout(a_out)
        # out (1, 3*embedding_dim)
        return q_out, a_out


if __name__ == "__main__":
    model = CnnModel(embedding_file="../data/word2vec/word2vec-300.iter5", embedding_dim=300, vocab_size=605)

    train_data_path = "../data/input/train.json"
    with open(train_data_path, 'r', encoding="utf-8") as load_j:
        train_data = json.load(load_j)
    train_questions = []
    train_answers = []
    train_labels = []
    for question in train_data.keys():
        for item in train_data[question]:
            train_labels.append(int(item[0]))
            train_questions.append(item[1])
            train_answers.append(item[2])
    train_questions = model.sentence_encode(train_questions)
    train_answers = model.sentence_encode(train_answers)
    print(train_questions.size())  # (50694, 1, 605)
    out = model(train_questions[0], train_answers[0])
    print(out)
    print(out.size())
    # # conv = nn.Conv2d(1, embedding_dim, (2, embedding_dim))
    # x = torch.zeros([1, 605], dtype=torch.long)
    # out = model(x)
