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
import nltk
from torchtext import data


# 测试用例的格式
class Testcase:
    def __init__(self, c_char, q_char, c_word, q_word):
        self.c_char = c_char
        self.q_char = q_char
        self.c_word = c_word
        self.q_word = q_word


def train(args, data):
    model = BiDAF(args, data.WORD.vocab.vectors)
    # print('**********before*********')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # An Adaptive Learning Rate Method
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    loss, last_epoch = 0, -1
    # 增加从 checkpoint 继续训练功能
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    criterion = nn.CrossEntropyLoss()
    # 记录模型和相关测试指标，并可视化
    writer = SummaryWriter(log_dir='logs/' + args.model_time)

    model.train()

    iterator = data.train_iter

    for i, batch in enumerate(iterator):
        current_epoch = int(iterator.epoch)
        if current_epoch == args.epoch:
            break
        if current_epoch > last_epoch:
            print('epoch:', current_epoch+1)
            # 保存checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'./checkpoints/epoch_{current_epoch}.pt')
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


def word_tokenize(str):
    return [token.replace("''", '"').replace("``", '"')
            for token in nltk.word_tokenize(str)]


def run_with_model(model, questions, contexts, word_vocab, char_vocab):
    """
    :param model:
    :param questions: list of question(str)
    :param contexts: list of context(str)
    :param word_vocab:
    :param char_vocab:
    :return:
    """

    with torch.no_grad():
        question_tokens = [[word_vocab[token] for token in word_tokenize(question)] for question in questions]
        context_tokens = [[word_vocab[token] for token in word_tokenize(context)] for context in contexts]
        q_word = (torch.tensor(question_tokens), torch.tensor(list(map(lambda question: len(question), question_tokens))))
        # q_word[0]: [batch_size, 8], q_word[1]: [batch_size]
        c_word = (torch.tensor(context_tokens), torch.tensor(list(map(lambda context: len(context), context_tokens))))
        question_char_len = max([len(token) for question in questions for token in word_tokenize(question)])
        context_char_len = max([len(token) for context in contexts for token in word_tokenize(context)])
        question_chars = [[[char_vocab[char] for char in token] + [0] * (question_char_len - len(token)) for token in word_tokenize(question)]
                          for question in questions]
        context_chars = [[[char_vocab[char] for char in token] + [0] * (context_char_len - len(token)) for token in word_tokenize(context)]
                         for context in contexts]
        # q_char: [batch_size, 8, 5]  c_char: [batch_size, 147, 11]
        q_char = torch.tensor(question_chars)
        c_char = torch.tensor(context_chars)
        test_examples = Testcase(c_char, q_char, c_word, q_word)
        p_start, p_end = model(test_examples)
        softmax = nn.Softmax(dim=1)
        start_idx = torch.argmax(softmax(p_start), dim=1)
        end_idx = torch.argmax(softmax(p_end), dim=1)
        # c_char.size(0): 测试用例的数量
        for i in range(c_char.size(0)):
            answer = (word_tokenize(contexts[i]))[start_idx[i]: end_idx[i]+1]
            print(answer)


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
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--learning_rate', default=0.5, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    # context_len参数必须与训练的模型保持一致
    parser.add_argument('--context_len', default=150, type=int)
    parser.add_argument('--resume', default=False, type=bool)

    args = parser.parse_args()
    # print('loading SQuAD data...')
    # data = SQuAD(args)
    with open('vocabs/char_vocab.pickle', 'rb') as handle:
        char_vocab = pickle.load(handle)
    with open('vocabs/pretrained_vectors.pickle', 'rb') as handle:
        pretrained_vectors = pickle.load(handle)
    with open('vocabs/word_vocab.pickle', 'rb') as handle:
        word_vocab = pickle.load(handle)

    # setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'char_vocab_size', len(char_vocab))
    setattr(args, 'model_time', strftime('%m_%d_%H_%M_%S', gmtime()))
    setattr(args, 'prediction_file', 'outputs/predictions_{}'.format(strftime('%m_%d_%H_%M_%S', gmtime())))
    setattr(args, 'dataset_file', 'inputs/dev-v1.1.json')
    setattr(args, 'checkpoint', './checkPoints/0308/epoch_4.pt')
    #
    # model = BiDAF(args, data.WORD.vocab.vectors)
    # model = BiDAF(args, pretrained_vectors)
    #
    # print('training start')
    # model = train(args, data)
    # if not os.path.exists('saved_models'):
    #     os.makedirs('saved_models')
    # torch.save(model.state_dict(), f'saved_models/BiDAF_{args.model_time}.pt')
    # model.load_state_dict(torch.load('saved_models/BiDAF_03_07_08_37_36.pt'))
    # questions = ["Where did Super Bowl 50 take place?", "Which NFL team won Super Bowl 50?"]
    # contexts = ["Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
    #             "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."]
    # run_with_model(model, questions=questions,
    #                contexts=contexts,
    #                word_vocab=word_vocab, char_vocab=char_vocab)
    # model.eval()

    # 从checkPoints 中读取model, 随后对model进行测试
    checkPoint = torch.load(args.checkpoint)
    model = BiDAF(args, pretrained_vectors)
    model.load_state_dict(checkPoint['model_state_dict'])
    dev_loss, dev_exact, dev_f1 = test(model, args, data)
    print(f'dev loss: {dev_loss: .3f}'
          f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')
    result = {
        "context_len=150, epoch=3, 0310": {
            "Dev loss": dev_loss,
            "Dev exact": dev_exact,
            "Dev F1": dev_f1,
        }
    }
    with open('./outputs/evaluation.json', 'a', encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)

