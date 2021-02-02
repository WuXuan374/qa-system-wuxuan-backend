import gensim
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import InteractLayer, MLP
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt


class CnnModel(nn.Module):
    def __init__(self, embedding_file, embedding_dim, vocab_size, save_path, word2idx,
                 n_in=20, n_hidden=128, n_out=2, filter_size=(2, 3, 4), dropout_rate=0.1):
        super(CnnModel, self).__init__()
        # 参数设置
        self.embedding_dim = embedding_dim
        self.num_of_filter = self.embedding_dim
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.embedding_file = embedding_file
        self.word2idx = word2idx
        # 加1是因为oov token的存在
        self.dict_size = len(self.word2idx.keys()) + 1
        print(self.dict_size)

        # 先使用随机生成的embedding
        self.embedding_layer = nn.Embedding(self.dict_size, self.embedding_dim)
        # self.embedding_layer, self.word2idx = self.read_word2vec()

        self.oov_index = self.dict_size-1
        self.num_feature_maps = len(filter_size) * self.num_of_filter
        self.input_size = n_in
        self.hidden_size = n_hidden
        self.out_size = n_out
        self.save_path = save_path

        # 模型
        # 多种conv model的集合，区别在于kernel_size不同： (2,3,4)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.num_of_filter,
                      kernel_size=(h, self.embedding_dim)) for h in self.filter_size
        ])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.interact_layer = InteractLayer(self.num_feature_maps, self.num_feature_maps, self.input_size)
        self.mlp = MLP(self.input_size, self.hidden_size, self.out_size)

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

    def conv_and_pool(self, x, conv):
        """

        :param x: (1, 1, vocab_size, embedding_dim)
        :param conv:
        :return:
        """
        # 进行卷积，对卷积的结果使用Relu激活函数（使得结果范围在[0,1)之间）
        x = torch.tanh(conv(x)).squeeze(3)
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
        # out (1, 3*embedding_dim) = (1, self.num_feature_maps)
        qa_vec = self.interact_layer(q_out, a_out)
        # qa_vec: (batch_size, self.input_size)
        prop, cate = self.mlp(qa_vec)
        return prop, cate


class ModelLoader:
    def __init__(self, model_path):
        # TODO: 暂时只支持从training data中找答案，后面改进
        self.model = torch.load(model_path)
        self.model.eval()
        self.train_data_path = "../data/input/train.json"
        with open(self.train_data_path, 'r', encoding="utf-8") as load_j:
            self.train_data = json.load(load_j)
        with open('../data/models/word2idx.pickle', 'rb') as handle:
            self.word2idx = pickle.load(handle)
        self.vocab_size = 605
        self.oov_index = len(self.word2idx.keys())

    def sentence_encode(self, sentences):
        """
        将文本表示成bag-of-words格式, 不在word2idx中的词，分配一个oov_index
        :param sentences: list of list, [["今天", "天气", "如何"],]
        :param vocab_size: 常量，所规定的的句子长度
        :param oov_index: oov-token index
        :param: word2idx: dict, e.g. {"今天": 1}
        :return:sentences_vector： (len(sentences), vocab_size), 每一行是一个句子对应的bag-of-words向量
        """
        sentences_vector = torch.zeros([len(sentences), self.vocab_size], dtype=torch.long)
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

    def get_answer_from_model(self, question_str):
        questions = []
        answers = []
        labels = []
        for item in self.train_data[question_str]:
            print(item)
            labels.append(int(item[0]))
            questions.append(item[1])
            answers.append(item[2])
        questions_matrix = self.sentence_encode(questions)  # (7445, vocab_size)
        answers_matrix = self.sentence_encode(answers)  # (7445, vocab_size)
        labels_matrix = torch.tensor(labels, dtype=torch.long)  # (7445, )
        pred_prop, pred_value = self.model(questions_matrix, answers_matrix)
        print(pred_prop)
        print(pred_value)


def sentence_encode(sentences, vocab_size, oov_index, word2idx):
    """
    将文本表示成bag-of-words格式, 不在word2idx中的词，分配一个oov_index
    :param sentences: list of list, [["今天", "天气", "如何"],]
    :param vocab_size: 常量，所规定的的句子长度
    :param oov_index: oov-token index
    :param: word2idx: dict, e.g. {"今天": 1}
    :return:sentences_vector： (len(sentences), vocab_size), 每一行是一个句子对应的bag-of-words向量
    """
    sentences_vector = torch.zeros([len(sentences), vocab_size], dtype=torch.long)
    for i in range(len(sentences)):
        sentence = sentences[i]
        # oov token: 0
        vector = torch.full((1, vocab_size), oov_index, dtype=torch.long)
        for j in range(min(len(sentence), vocab_size)):
            word = sentence[j]
            if word in word2idx.keys():
                vector[0, j] = word2idx[word]
        sentences_vector[i, :] = vector
    return sentences_vector


def accuracy(predicted_labels, yb):
    """
    比较模型预测的标签和实际的标签， 计算模型的准确率
    :param: predicted_labels: (batch_size, )
    :param: yb: (batch_size, )
    :return:
    """
    return (predicted_labels == yb).float().mean()


def get_data_by_batch(train_ds, val_ds, bs):
    """
    按照batch从数据集中取数据
    :param train_ds: dataset
    :param val_ds: dataset
    :param bs: num, batch_size
    :return: DataLoader
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds, batch_size=bs, shuffle=True)
    )


def fit(epochs, model, opt, loss_func, train_dl, val_dl):
    """
    :param epochs: num
    :param model:
    :param opt: optimizer
    :param loss_func: cost function
    :param train_dl: DataLoader
    :param val_dl: DataLoader
    :return: loss_train, list
    :return: loss_valid, list
    """
    # 记录train dataset, validation dataset的loss,用于作图
    loss_train = []
    loss_valid = []
    for epoch in range(epochs):
        # 只是告知模型，正处于训练状态
        model.train()
        epoch_loss = 0
        # 按batch从DataLoader中取数据
        for question_batch, answer_batch, label_batch in train_dl:
            # pred: (800,)
            prop, pred = model(question_batch, answer_batch)
            print('label', label_batch.size())
            loss = loss_func(prop.float(), label_batch)
            with torch.no_grad():
                epoch_loss += loss
            # 根据loss进行反向传播
            loss.backward()
            # 更新所有的参数
            opt.step()
            # 梯度置0
            opt.zero_grad()
        model.eval()
        loss_train.append(epoch_loss/len(train_dl))
        with torch.no_grad():
            val_loss = sum(loss_func(model(qb, ab)[0], lb) for qb, ab, lb in val_dl)
            loss_valid.append(val_loss/len(val_dl))
        print(epoch, val_loss/len(val_dl))
    torch.save(model, model.save_path)
    # torch.save(model.state_dict(), model.save_path)
    return loss_train, loss_valid


def draw_loss(loss_train, loss_valid, epochs):
    x = range(1, epochs+1)
    y1 = loss_train
    y2 = loss_valid
    plt.plot(x, y1, 'o-')
    plt.plot(x, y2, '.-')
    plt.title('train loss VS validation loss')
    plt.show()


def train(model, optimizer, train_ds, val_ds, bs, epochs):
    train_dl, val_dl = get_data_by_batch(train_ds, val_ds, bs)
    loss_func = nn.CrossEntropyLoss()
    loss_train, loss_validation = fit(epochs, model, optimizer, loss_func, train_dl, val_dl)
    draw_loss(loss_train, loss_validation, epochs)
    train_accu = sum(accuracy(model(qb, ab)[1], lb) for qb, ab, lb in train_dl)/len(train_dl)
    print('train ', train_accu)
    val_accu = sum(accuracy(model(qb, ab)[1], lb) for qb, ab, lb in val_dl) / len(val_dl)
    print('validation ', val_accu)
    return train_accu, val_accu


def test(train_ds, val_ds, bs, model_path):
    train_dl, val_dl = get_data_by_batch(train_ds, val_ds, bs)
    model = torch.load(model_path)
    print(model.parameters())
    model.eval()
    train_accu = sum(accuracy(model(qb, ab)[1], lb) for qb, ab, lb in train_dl)/len(train_dl)
    print('train ', train_accu)
    val_accu = sum(accuracy(model(qb, ab)[1], lb) for qb, ab, lb in val_dl) / len(val_dl)
    print('validation ', val_accu)
    return train_accu, val_accu


if __name__ == "__main__":
    # 参数定义
    vocab_size = 605
    embedding_dim = 100
    n_in = 20
    n_hidden = 128
    n_out = 2
    lr = 1e-5
    bs = 800
    epochs = 7
    with open('../data/models/word2idx.pickle', 'rb') as handle:
        word2idx = pickle.load(handle)
    oov_index = len(word2idx.keys())  # 635963
    # word2idx需要提前保存，不然使用模型时要花很长时间加载
    # with open('../data/models/word2idx.pickle', 'wb') as handle:
    #     pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = CnnModel(embedding_file="../data/word2vec/word2vec-300.iter5",
                     embedding_dim=embedding_dim, vocab_size=vocab_size,
                     save_path="../data/models/model_CNN_epochs=7_0202.pth",
                     word2idx=word2idx, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 数据读取
    train_data_path = "../data/input/train.json"
    validation_data_path = "../data/input/validation.json"
    with open(train_data_path, 'r', encoding="utf-8") as load_j:
        train_data = json.load(load_j)
    with open(validation_data_path, 'r', encoding="utf-8") as load_j:
        validation_data = json.load(load_j)
    # 训练集数据
    train_questions = []
    train_answers = []
    train_labels = []
    for question in train_data.keys():
        for item in train_data[question]:
            train_labels.append(int(item[0]))
            train_questions.append(item[1])
            train_answers.append(item[2])
    train_questions = sentence_encode(train_questions, vocab_size, oov_index, word2idx)  # (50694, vocab_size)
    train_answers = sentence_encode(train_answers, vocab_size, oov_index, word2idx)  # (50694, vocab_size)
    train_labels = torch.tensor(train_labels, dtype=torch.long)  # (50964, )

    # 验证集数据
    val_questions = []
    val_answers = []
    val_labels = []
    for question in validation_data.keys():
        for item in validation_data[question]:
            val_labels.append(int(item[0]))
            val_questions.append(item[1])
            val_answers.append(item[2])
    val_questions = sentence_encode(val_questions, vocab_size, oov_index, word2idx)  # (7445, vocab_size)
    val_answers = sentence_encode(val_answers, vocab_size, oov_index, word2idx)  # (7445, vocab_size)
    val_labels = torch.tensor(val_labels, dtype=torch.long)  # (7445, )

    # TensorDataset: dataset wrapping, 方便分批从数据集中取数据
    train_dataset = TensorDataset(train_questions, train_answers, train_labels)
    val_dataset = TensorDataset(val_questions, val_answers, val_labels)

    # 训练
    train_accu, validation_accu = train(model, optimizer, train_dataset, val_dataset, bs, epochs)

    # # 读取保存的模型，测试模型能否使用
    # train_accu, val_accu = test(train_dataset, val_dataset, bs,
    #                             model_path="../data/models/model_CNN_epochs=1_0202.pth")

    # model_loader = ModelLoader('../data/models/model_CNN_epochs=1_0202.pth')
    # model_loader.get_answer_from_model("大叻大学的越语是什么，在什么地方？")
    # prop: (5,2), 在0/1标签上的概率分布
    # cate: (5,) [1,1,0,1,0] 所预测的标签
