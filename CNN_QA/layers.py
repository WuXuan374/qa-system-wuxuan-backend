import torch
import torch.nn as nn
from torch.autograd import Variable


class InteractLayer(nn.Module):
    def __init__(self, q_dim, a_dim, dim):
        """
        计算question_vec和answer_vec之间的相似度
        计算公式: question_vector * weights * (answer_vector)T
        weights 是在训练过程中不断优化
        :param q_dim: pool之后，question_vec的维度。 len(filter_size) * self.num_of_filter
        :param a_dim: answer_vec的维度
        PS: q_dim == a_dim
        :param dim: 三者相乘后结果的维度。 三者相乘后的结果作为Multi-layer perception的输入层。
        """
        super(InteractLayer, self).__init__()
        # Variable：具备自动求导功能，整合了反向传播的相关实现
        self.weights = Variable(torch.randn(q_dim, dim, a_dim) * 0.05)
        self.dim = dim
        self.input_size = q_dim

    def forward(self, question_vec, answer_vec):
        # torch.mm: 矩阵乘法
        # question_vec: (batch_size, 1, q_dim)
        qw = torch.mm(question_vec, self.weights.view(self.input_size, -1)).view(-1, self.dim, self.input_size)
        # qw: (batch_size, dim, a_dim)
        # answer_vec(unsqueeze 后）: (batch_size, a_dim,1 )
        qwa = torch.bmm(qw, torch.unsqueeze(answer_vec, 2))
        # bmm: (batch_size, dim, a_dim) * (batch_size, a_dim, 1) -> (batch_size, dim, 1)
        qa_vec = qwa.view(-1, self.dim)
        # qa_vec: (batch_size, dim)
        return qa_vec


class MLP(nn.Module):
    """
    multi-layer perceptron
    三层架构： input layer, hidden layer, softmax output layer
    """
    def __init__(self, n_in, n_hidden, n_out):
        """
        预测问题-答案对所对应的标签
        :param n_in: input layer dimension
        :param n_hidden: hidden layer dimension
        :param n_out: softmax layer dimesion. 本模型为2 --> 0/1这两个label
        """
        super(MLP, self).__init__()
        # softmax 层计算不同label的概率分布
        # nn.Linear: 全连接层，实现y=ax+b, n_in为x维度, n_hidden为y维度
        self.mlp = nn.Sequential(nn.Linear(n_in, n_hidden),
                                 nn.Tanh(),
                                 nn.Linear(n_hidden, n_out),
                                 nn.Softmax(dim=1)
                                 )

    def forward(self, inputs):
        pred_prop = self.mlp(inputs)
        # dim=1: 按行取最大值
        _, pred_value = torch.max(pred_prop, dim=1)
        return pred_prop, pred_value

