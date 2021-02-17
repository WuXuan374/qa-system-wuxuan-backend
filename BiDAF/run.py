import argparse
from BiDAF.data import SQuAD
from BiDAF.model import BiDAF
import evaluation
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from time import gmtime, strftime
import os
import json
import pickle


def train(args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(args, data.WORD.vocab.vectors).to(device)
    # print('**********before*********')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # An Adaptive Learning Rate Method
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    # 记录模型和相关测试指标，并可视化
    writer = SummaryWriter(log_dir='logs/' + args.model_time)

    model.train()

    iterator = data.train_iter
    loss, last_epoch = 0, -1

    for i, batch in enumerate(iterator):
        current_epoch = int(iterator.epoch)
        if current_epoch == args.epoch:
            break
        if current_epoch > last_epoch:
            print('epoch:', current_epoch+1)
        print('batch_index', i)
        last_epoch = current_epoch

        # pStart [60, 376] [batch_size, c_len]
        pStart, pEnd = model(batch)
        # batch.start_idx [60] [batch_size]

        optimizer.zero_grad()
        # 分别对答案的起始位置，结束位置计算交叉熵
        batch_loss = criterion(pStart, batch.start_idx) + criterion(pEnd, batch.end_idx)
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_exact, dev_f1 = test(model, args, data)
            c = i+1

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('exact_match/dev', dev_exact, c)
            writer.add_scalar('f1/dev', dev_f1, c)
            print(f'train loss:{loss: .3f} / dev loss: {dev_loss: .3f}'
                  f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

        loss = 0
        model.train()

    writer.close()
    return model


def test(model, args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            pStart, pEnd = model(batch)
            batch_loss = criterion(pStart, batch.start_idx) + criterion(pEnd, batch.end_idx)
            loss += batch_loss.item()

            batch_size, c_len = pStart.size()

            # # dim=1: 对每一行元素进行softmax运算，每一行的和为1
            # ls = nn.LogSoftmax(dim=1)
            # # mask: (batch_size, c_len, c_len)
            # mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            # # score: (batch_size, c_len, c_len)
            # score = (ls(pStart).unsqueeze(2) + ls(pEnd).unsqueeze(1)) + mask
            # score, s_idx = score.max(dim=1)
            # score, e_idx = score.max(dim=1)
            # s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            # dim=1: 对每一行元素进行softmax运算，每一行的和为1
            softmax = nn.Softmax(dim=1)
            # argmax(dim=1): 按行取最大值的下标
            start_idx = torch.argmax(softmax(pStart), dim=1)
            end_idx = torch.argmax(softmax(pEnd), dim=1)

            for i in range(batch_size):
                id = batch.id[i]
                answer = batch.c_word[0][i][start_idx[i]:end_idx[i] + 1]
                answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
                answers[id] = answer
    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluation.main(args)
    return loss, results['EM'], results['F1']


def run_with_model(model, question, context, word_vocab, char_vocab):
    data = {}
    print(char_vocab)
    print(word_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="train-v1.1.json")
    parser.add_argument('--dev-file', default="dev-v1.1.json")
    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--train_batch_size', default=50, type=int)
    parser.add_argument('--dev_batch_size', default=100, type=int)
    parser.add_argument('--char_dim', default=8, type=int)
    parser.add_argument('--char_channel_size', default=100, type=int)
    parser.add_argument('--char_channel_width', default=5, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--learning_rate', default=0.5, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--context_len', default=50, type=int)

    args = parser.parse_args()
    #
    # print('loading SQuAD data...')
    # data = SQuAD(args)
    with open('vocabs/char_vocab.pickle', 'rb') as handle:
        char_vocab = pickle.load(handle)
    with open('vocabs/pretrained_vectors.pickle', 'rb') as handle:
        pretrained_vectors = pickle.load(handle)

    # setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'char_vocab_size', len(char_vocab))
    setattr(args, 'model_time', strftime('%m_%d_%H_%M_%S', gmtime()))
    setattr(args, 'prediction_file', 'outputs/predictions_{}'.format(strftime('%m_%d_%H_%M_%S', gmtime())))
    setattr(args, 'dataset_file', 'inputs/dev-v1.1.json')
    #
    # model = BiDAF(args, data.WORD.vocab.vectors)
    model = BiDAF(args, pretrained_vectors)
    #
    # print('training start')
    # model = train(args, data)
    # if not os.path.exists('saved_models'):
    #     os.makedirs('saved_models')
    # torch.save(model.state_dict(), f'saved_models/BiDAF_{args.model_time}.pt')
    model.load_state_dict(torch.load('saved_models/BiDAF_02_17_14_59_12.pt'))
    model.eval()

