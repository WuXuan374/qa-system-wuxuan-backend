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
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Loss(x, class) = - alpha * (1-softmax(x)[class])^gamma * log(softmax(x)[class])
    """
    def __init__(self, class_num, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        # if alpha is None:
        #     # alpha: (class_num, 1)
        #     self.alpha = Variable(torch.ones(class_num, 1))
        # else:
        #     if isinstance(alpha, Variable):
        #         self.alpha = alpha
        #     else:
        #         self.alpha = Variable(alpha)
        self.alpha = torch.Tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, inputs, targets):
        """
        计算预测值和实际值之间的focal loss
        :param inputs: 预测值, (batch_size, class_num)
        :param targets: 实际值, (batch_size, )
        :return:
        """
        # P (batch_size, class_num)
        p = F.softmax(inputs)

        # 创建一个类型、维度与inputs一致的全0矩阵
        class_mask = inputs.data.new(p.size()).fill_(0)
        class_mask = Variable(class_mask)
        # ids: (batch_size, 1)
        ids = targets.view(-1, 1)
        # scatter_(dim, index, src)
        # 原本targets: （batch_size, ), 转换成 (batch_size, class_num) --> 相应class处为1，则代表实际的标签
        # targets：[1, 0, 0, 1] --> class_mask: [[0,1], [1,0], [1,0], [0,1]
        # class_mask: (batch_size, class_num)
        class_mask.scatter_(1, ids.data, 1.)

        # before: alpha(2, 1)
        alpha = self.alpha.type_as(inputs.data)
        alpha = alpha.gather(0, targets.data.view(-1))
        # after: alpha(batch_size, 1)

        # P (batch_size, class_num), class_mask (batch_size, class_num)
        # 预测值和实际值对应位相乘，并对行求和
        # probs: (batch_size, 1)
        probs = (p*class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # batch_loss (batch_size, 1)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # loss: num, 取得的平均值
        loss = batch_loss.mean()
        return loss


# class FocalLoss:
#     def __init__(self, gamma=2., alpha=.25):
#         self.gamma = gamma
#         self.alpha = alpha
#
#     def focal_loss(self, y_pred, y_target):
#         """
#
#         :param y_pred: (batch_size, class_num)
#         :param y_target: (batch_size)
#         :return:
#         """
#         # 创建一个类型、维度与y_target一致的全0矩阵
#         class_mask = y_target.data.new(y_pred.size()).fill_(0)
#         class_mask = Variable(class_mask)
#         # ids: (batch_size, 1)
#         ids = y_target.view(-1, 1)
#         # scatter_(dim, index, src)
#         # 原本targets: （batch_size, ), 转换成 (batch_size, class_num) --> 相应class处为1，则代表实际的标签
#         # targets：[1, 0, 0, 1] --> class_mask: [[0,1], [1,0], [1,0], [0,1]
#         # class_mask: (batch_size, class_num)
#         class_mask.scatter_(1, ids.data, 1.)
#         pt_1 = torch.where(class_mask == 1, y_pred, torch.ones(y_pred.size()))
#         print(pt_1.size(), pt_1)
#         pt_0 = torch.where(class_mask == 0, y_pred, torch.zeros(y_pred.size()))
#         print(pt_0.size(), pt_0)
#         left = -(self.alpha * torch.pow(1. - pt_0, self.gamma) * torch.log(pt_0.float()))
#         print(left)


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
        # self.dict_size = len(self.word2idx.keys()) + 1
        # print(self.dict_size)
        # with open('../data/models/word2idx.pickle', 'wb') as handle:
        #     pickle.dump(self.word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        x = torch.relu(conv(x)).squeeze(3)
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
        self.train_data_path = "../data/input/train_2017.json"
        with open(self.train_data_path, 'r', encoding="utf-8") as load_j:
            self.train_data = json.load(load_j)
        self.validation_data_path = "../data/input/validation.json"
        with open(validation_data_path, 'r', encoding="utf-8") as load_j:
            self.validation_data = json.load(load_j)
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
        for question in self.train_data.keys():
            questions = []
            answers = []
            labels = []
            for item in self.train_data[question]:
                labels.append(int(item[0]))
                questions.append(item[1])
                answers.append(item[2])
            questions_matrix = self.sentence_encode(questions)  # (7445, vocab_size)
            answers_matrix = self.sentence_encode(answers)  # (7445, vocab_size)
            labels_matrix = torch.tensor(labels, dtype=torch.long)  # (7445, )
            pred_prop, pred_value = self.model(questions_matrix, answers_matrix)
            print(labels_matrix)
            print(pred_prop)
            print(pred_value)
        # questions_matrix = self.sentence_encode(questions)  # (7445, vocab_size)
        # answers_matrix = self.sentence_encode(answers)  # (7445, vocab_size)
        # labels_matrix = torch.tensor(labels, dtype=torch.long)  # (7445, )
        # pred_prop, pred_value = self.model(questions_matrix, answers_matrix)
        # print(labels_matrix)
        # # print(pred_prop)
        # print(pred_value)

    def evaluation(self, data):
        """
        计算用于评估模型效果的相关指标
        :param data: from json.load
        :return: mrr(num)
        :return: acc(num): 针对每一个问题，对模型预测的props进行排序，取前三者作为候选答案。如果候选答案中存在正确答案，则acc_sum+=1, mrr_sum+= 1/(index+1)
        :return: true_answer_acc: 真实标签为"1"处，如果预测标签为"1", 则true_label_acc_sum+=1
        """
        question_num = len(data.keys())
        mrr_sum = 0
        acc_sum = 0
        true_label_acc_sum = 0
        # for question_str in data.keys():
        for question_str in list(data.keys())[:1000]:
            print(question_str)
            labels = list(map(lambda item: int(item[0]), data[question_str]))
            questions = list(map(lambda item: item[1], data[question_str]))
            answers = list(map(lambda item: item[2], data[question_str]))
            answers_matrix = self.sentence_encode(answers)
            questions_matrix = self.sentence_encode(questions)
            labels_matrix = torch.tensor(labels, dtype=torch.long)
            pred_prop, pred_value = self.model(questions_matrix, answers_matrix)
            print(pred_prop)
            if torch.equal(torch.nonzero(labels_matrix), torch.nonzero(pred_value)):
                true_label_acc_sum += 1
            # enumerate(answers): [(0, ["北京大学",])], x[0]: index, pred_prop[x[0]][1]: 预测标签为1的softmax概率
            candidate_answers = sorted(enumerate(answers), key=lambda x: float(pred_prop[x[0]][1]), reverse=True)[:3]
            print(candidate_answers)
            for index, answer in candidate_answers:
                print(index, labels[index], answer)
                if labels[index] == 1:
                    mrr_sum += 1 / (index + 1)
                    acc_sum += 1
                    break
        mrr = mrr_sum / question_num
        acc = acc_sum / question_num
        true_answer_acc = true_label_acc_sum / question_num
        return mrr, acc, true_answer_acc


def sentence_encode(sentences, vocab_size, oov_index, word2idx):
    """
    将文本表示成bag-of-words格式, 不在word2idx中的词，分配一个oov_index
    :param sentences: list of list, [["今天", "天气", "如何"],]
    :param vocab_size: 常量，所规定的的句子长度
    :param oov_index: oov-token index
    :param: word2idx: dict, e.g. {"今天": 1}
    :return:sentences_vector： (len(sentences), vocab_size), 每一行是一个句子对应的bag-of-words向量
    """
    # sentences_vector = torch.zeros([len(sentences), vocab_size], dtype=torch.long)
    sentences_vector = np.zeros([len(sentences), vocab_size], dtype=np.long)
    for i in range(len(sentences)):
        sentence = sentences[i]
        # oov token: 0
        # vector = torch.full((1, vocab_size), oov_index, dtype=torch.long)
        vector = np.full((1, vocab_size), oov_index, dtype=np.long)
        for j in range(min(len(sentence), vocab_size)):
            word = sentence[j]
            if word in word2idx.keys():
                vector[0, j] = word2idx[word]
        sentences_vector[i, :] = vector
    return sentences_vector


def over_sample_data(question, answers, labels, vocab_size):
    """
    由于数据集存在数据不均衡的问题：标签为0的数据: 标签为1的数据 = 15:1
    我们需要对数据进行均衡，避免影响训练效果
    返回的数据都是torch tensor, 满足模型训练的需要
    :param question: numpy array (50694, vocab_size)
    :param answers: numpy array (50694, vocab_size)
    :param labels: list, len=50694
    :return: questions, answers, labels: torch.tensor
    """
    # 需要对问题和答案数组进行拼接，因为fit_sample只接受x,y这两个输入
    x_train = np.concatenate((question, answers), axis=1)
    sm = SMOTE(random_state=2)
    x_train, labels = sm.fit_sample(x_train, labels)
    questions = x_train[..., 0:vocab_size]
    answers = x_train[..., vocab_size:]
    questions = torch.tensor(questions, dtype=torch.long)
    answers = torch.tensor(answers, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return questions, answers, labels


# def accuracy(predicted_labels, yb):
#     """
#     比较模型预测的标签和实际的标签， 计算模型的准确率
#     :param: predicted_labels: (batch_size, )
#     :param: yb: (batch_size, )
#     :return:
#     """
#     return (predicted_labels == yb).float().mean()
def evaluation(data, vocab_size, oov_index, word2idx, model):
    """
    计算用于评估模型效果的相关指标
    :param data: from json.load
    :param vocab_size:
    :param oov_index:
    :param word2idx:
    :param model:
    :return: mrr(num)
    :return: acc(num): 针对每一个问题，对模型预测的props进行排序，取前三者作为候选答案。如果候选答案中存在正确答案，则acc_sum+=1, mrr_sum+= 1/(index+1)
    :return: true_answer_acc: 真实标签为"1"处，如果预测标签为"1", 则true_label_acc_sum+=1
    """
    question_num = len(data.keys())
    mrr_sum = 0
    acc_sum = 0
    true_label_acc_sum = 0
    for question_str in data.keys():
        labels = list(map(lambda item: int(item[0]), data[question_str]))
        questions = list(map(lambda item: item[1], data[question_str]))
        answers = list(map(lambda item: item[2], data[question_str]))
        answers_matrix = torch.tensor(sentence_encode(answers, vocab_size, oov_index, word2idx), dtype=torch.long)
        questions_matrix = torch.tensor(sentence_encode(questions, vocab_size, oov_index, word2idx), dtype=torch.long)
        labels_matrix = torch.tensor(labels, dtype=torch.long)
        pred_prop, pred_value = model(questions_matrix, answers_matrix)
        if torch.equal(torch.nonzero(labels_matrix), torch.nonzero(pred_value)):
            true_label_acc_sum += 1
        # enumerate(answers): [(0, ["北京大学",])], x[0]: index, pred_prop[x[0]][1]: 预测标签为1的softmax概率
        candidate_answers = sorted(enumerate(answers), key=lambda x: float(pred_prop[x[0]][1]), reverse=True)[:3]
        for index, answer in candidate_answers:
            print(index, answer)
            if labels[index] == 1:
                mrr_sum += 1/(index+1)
                acc_sum += 1
                break
    mrr = mrr_sum/question_num
    acc = acc_sum/question_num
    true_answer_acc = true_label_acc_sum/question_num
    return mrr, acc, true_answer_acc


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
            # pred: (batch_size,)
            prop, pred = model(question_batch, answer_batch)
            print('label', label_batch.size())
            loss = loss_func(prop.float(), label_batch)
            # loss = loss_func.focal_loss(prop.float(), label_batch)
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


def train(model, optimizer, train_ds, val_ds, bs, epochs, train_data, validation_data, vocab_size, oov_index, word2idx):
    train_dl, val_dl = get_data_by_batch(train_ds, val_ds, bs)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = FocalLoss(class_num=2)
    loss_train, loss_validation = fit(epochs, model, optimizer, loss_func, train_dl, val_dl)
    draw_loss(loss_train, loss_validation, epochs)
    train_mrr, train_acc, train_true_answer_acc = evaluation(train_data, vocab_size, oov_index, word2idx, model)
    val_mrr, val_acc, val_true_answer_acc = evaluation(validation_data, vocab_size, oov_index, word2idx, model)
    print('train mrr ', train_mrr)
    print('train_acc ', train_acc)
    print('train_true_answer_acc', train_true_answer_acc)
    print('val_mrr ', val_mrr)
    print('val_acc ', val_acc)
    print('val_true_answer_acc ', val_true_answer_acc)
    return train_mrr, train_acc, train_true_answer_acc, val_mrr, val_acc, val_true_answer_acc


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


def main(train_path, val_path, model_save_path, lang="zh", smote=False):
    vocab_size = 30
    embedding_dim = 300
    n_in = 40
    n_hidden = 256
    n_out = 2
    lr = 2e-5
    bs = 400
    epochs = 3

    if lang == "zh":
        with open('../data/models/word2idx.pickle', 'rb') as handle:
            word2idx = pickle.load(handle)
    else:
        with open('../data/models/word2idx_en.pickle', 'rb') as handle:
            word2idx = pickle.load(handle)
    oov_index = len(word2idx.keys())
    print('oov_index', oov_index)

    # train_data_path = "./input/TrecQA_train.json"
    # validation_data_path = "./input/TrecQA_dev.json"
    train_data_path = train_path
    validation_data_path = val_path
    with open(train_data_path, 'r', encoding="utf-8") as load_j:
        train_data = json.load(load_j)
    with open(validation_data_path, 'r', encoding="utf-8") as load_j:
        validation_data = json.load(load_j)
    model = CnnModel(embedding_file=
                        "../data/word2vec/gensim_glove.6B.300d.txt"
                        if lang == "en"
                        else "../data/word2vec/word2vec-300.iter5",
                     embedding_dim=embedding_dim, vocab_size=vocab_size,
                     # save_path="../data/models/model_CNN_TrecQA_train_epochs5.pth",
                     save_path=model_save_path,
                     word2idx=word2idx, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_questions = []
    train_answers = []
    train_labels = []
    for question in train_data.keys():
        for item in train_data[question]:
            train_labels.append(int(item[0]))
            train_questions.append(item[1])
            train_answers.append(item[2])
    train_questions = sentence_encode(train_questions, vocab_size, oov_index, word2idx)
    train_answers = sentence_encode(train_answers, vocab_size, oov_index, word2idx)

    # 正负样本平均
    if smote:
        train_questions, train_answers, train_labels = \
                over_sample_data(train_questions, train_answers, train_labels, vocab_size)
    else:
        train_questions = torch.tensor(train_questions, dtype=torch.long)
        train_answers = torch.tensor(train_answers, dtype=torch.long)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    print('train_answers', train_answers.size())

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

    val_questions = torch.tensor(val_questions, dtype=torch.long)
    val_answers = torch.tensor(val_answers, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    print('val_answers', val_answers.size())
    # TensorDataset: dataset wrapping, 方便分批从数据集中取数据
    train_dataset = TensorDataset(train_questions, train_answers, train_labels)
    val_dataset = TensorDataset(val_questions, val_answers, val_labels)
    # 训练
    train_mrr, train_acc, train_true_answer_acc, val_mrr, val_acc, val_true_answer_acc = \
        train(model, optimizer, train_dataset, val_dataset, bs, epochs, train_data, validation_data, vocab_size,
              oov_index, word2idx)

if __name__ == "__main__":
    # # 参数定义
    # vocab_size = 15
    # embedding_dim = 300
    # n_in = 40
    # n_hidden = 256
    # n_out = 2
    # lr = 2e-5
    # bs = 400
    # epochs = 5
    # with open('../data/models/word2idx.pickle', 'rb') as handle:
    #     word2idx = pickle.load(handle)
    # oov_index = len(word2idx.keys())  # 70000
    # # word2idx需要提前保存，不然使用模型时要花很长时间加载
    # # with open('../data/models/word2idx.pickle', 'wb') as handle:
    # #     pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # 数据读取
    # train_data_path = "../data/input/train_2017.json"
    # validation_data_path = "../data/input/validation.json"
    # with open(train_data_path, 'r', encoding="utf-8") as load_j:
    #     train_data = json.load(load_j)
    # with open(validation_data_path, 'r', encoding="utf-8") as load_j:
    #     validation_data = json.load(load_j)
    #
    # model = CnnModel(embedding_file="../data/word2vec/word2vec-300.iter5",
    #                  embedding_dim=embedding_dim, vocab_size=vocab_size,
    #                  save_path="../data/models/model_CNN_train2017_epochs5.pth",
    #                  word2idx=word2idx, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # # 训练集数据
    # train_questions = []
    # train_answers = []
    # train_labels = []
    # for question in train_data.keys():
    #     for item in train_data[question]:
    #         train_labels.append(int(item[0]))
    #         train_questions.append(item[1])
    #         train_answers.append(item[2])
    # # 查看训练集中句子的长度
    # # question_len = map(lambda question: len(question), train_questions)
    # # print(Counter(question_len))
    # # answer_len = map(lambda answer: len(answer), train_answers)
    # # print(Counter(answer_len))
    # train_questions = sentence_encode(train_questions, vocab_size, oov_index, word2idx)  # (50694, vocab_size)
    # train_answers = sentence_encode(train_answers, vocab_size, oov_index, word2idx)  # (50694, vocab_size)
    #
    # # with SMOTE
    # # train_questions: torch.Size([95334, vocab_size])
    # # train_answers: torch.Size([95334, vocab_size])
    # # train_labels: torch.Size([95334])
    # # train_questions, train_answers, train_labels = \
    # #     over_sample_data(train_questions, train_answers, train_labels, vocab_size)
    #
    # # without SMOTE
    # train_questions = torch.tensor(train_questions, dtype=torch.long)
    # train_answers = torch.tensor(train_answers, dtype=torch.long)
    # train_labels = torch.tensor(train_labels, dtype=torch.long)
    # print('train_answers', train_answers.size())
    #
    # # 验证集数据
    # val_questions = []
    # val_answers = []
    # val_labels = []
    # for question in validation_data.keys():
    #     for item in validation_data[question]:
    #         val_labels.append(int(item[0]))
    #         val_questions.append(item[1])
    #         val_answers.append(item[2])
    # val_questions = sentence_encode(val_questions, vocab_size, oov_index, word2idx)  # (7445, vocab_size)
    # val_answers = sentence_encode(val_answers, vocab_size, oov_index, word2idx)  # (7445, vocab_size)
    # # val_questions: torch.Size([14022, vocab_size])
    # # val_answers: torch.Size([14022, vocab_size])
    # # val_labels: torch.Size([14022])
    #
    # # with SMOTE
    # # val_questions, val_answers, val_labels = over_sample_data(val_questions, val_answers, val_labels, vocab_size)
    #
    # # without SMOTE
    # val_questions = torch.tensor(val_questions, dtype=torch.long)
    # val_answers = torch.tensor(val_answers, dtype=torch.long)
    # val_labels = torch.tensor(val_labels, dtype=torch.long)
    # print('val_answers', val_answers.size())
    # # TensorDataset: dataset wrapping, 方便分批从数据集中取数据
    # train_dataset = TensorDataset(train_questions, train_answers, train_labels)
    # val_dataset = TensorDataset(val_questions, val_answers, val_labels)
    #
    # # 训练
    # train_mrr, train_acc, train_true_answer_acc, val_mrr, val_acc, val_true_answer_acc = \
    #     train(model, optimizer, train_dataset, val_dataset, bs, epochs, train_data, validation_data, vocab_size, oov_index, word2idx)
    #
    # # 读取保存的模型，测试模型能否使用
    # # train_accu, val_accu = test(train_dataset, val_dataset, bs,
    # #                             model_path="../data/models/model_CNN_epochs=1_0202.pth")
    #
    # # model_loader = ModelLoader('../data/models/model_CNN_focalloss_epochs=5.pth')
    # # # model_loader.get_answer_from_model("大叻大学的越语是什么，在什么地方？")
    # # train_mrr, train_acc, train_true_answer_acc = model_loader.evaluation(model_loader.train_data)
    # # val_mrr, val_acc, val_true_answer_acc = model_loader.evaluation(model_loader.validation_data)
    # # print('train mrr ', train_mrr)
    # # print('train_acc ', train_acc)
    # # print('train_true_answer_acc', train_true_answer_acc)
    # # print('val_mrr ', val_mrr)
    # # print('val_acc ', val_acc)
    # # print('val_true_answer_acc ', val_true_answer_acc)
    #
    # # prop: (5,2), 在0/1标签上的概率分布
    # # cate: (5,) [1,1,0,1,0] 所预测的标签
    main(
        train_path="../data/input/train_2017.json",
        val_path="../data/input/validation.json",
        model_save_path="../data/models/model_CNN_train2017_smote_epochs5.pth",
        lang="zh",
        smote=True
    )
