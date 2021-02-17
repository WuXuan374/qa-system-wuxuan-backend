import json
import os
import nltk
import argparse
import torch
import pickle
from torchtext import data
from torchtext.vocab import GloVe


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"')
            for token in nltk.word_tokenize(tokens)]


class SQuAD:
    def __init__(self, args):
        prefix = './inputs'
        dataset_prefix = prefix + '/torchtext/'
        train_examples_path = dataset_prefix + 'train_examples.pt'
        dev_examples_path = dataset_prefix + 'dev_examples.pt'

        # e.g. ./inputs/processed_train.json
        if not os.path.exists('{}/{}_processed'.format(prefix, args.train_file)):
            self.preprocess('{}/{}'.format(prefix, args.train_file))
        if not os.path.exists('{}/{}_processed'.format(prefix, args.dev_file)):
            self.preprocess('{}/{}'.format(prefix, args.dev_file))

        # 1. Set up Fields
        # is_target: is target variable?
        self.RAW = data.RawField(is_target=False)
        self.CHAR_NESTING = data.Field(batch_first=True, lower=True, tokenize=list)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        # include_lengths return (padded_minibatch, lengths of each example)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True,
                               include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        # 用于从json文件中读取数据
        dict_fields = {'id': ('id', self.RAW),
                       'start_idx': ('start_idx', self.LABEL),
                       'end_idx': ('end_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        # 用于生成自己的数据集
        list_fields = [('id', self.RAW), ('start_idx', self.LABEL), ('end_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        # 数据已经处理好，则直接读取
        if os.path.exists(train_examples_path) and os.path.exists(dev_examples_path):
            print('loading')
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print('building')
            # 2. Make splits for data
            self.train, self.dev = data.TabularDataset.splits(
                path=prefix,
                train='{}_processed'.format(args.train_file),
                validation='{}_processed'.format(args.dev_file),
                format='json',
                fields=dict_fields,
            )

            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        # 出于效率考虑，删除过长的context
        # len 19649
        self.train.examples = [e for e in self.train.examples if len(e.c_word) < args.context_len]
        self.dev.examples = [e for e in self.dev.examples if len(e.c_word) < 100]

        # 3. build vocabulary
        print('building vocab')
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))
        if not os.path.exists('vocabs/char_vocab.pickle'):
            print('char_vocab')
            with open('vocabs/char_vocab.pickle', 'wb') as handle:
                pickle.dump(self.CHAR.vocab, handle)
        if not os.path.exists('vocabs/word_vocab.pickle'):
            print('word_vocab')
            with open('vocabs/word_vocab.pickle', 'wb') as handle:
                pickle.dump(self.WORD.vocab, handle)
        if not os.path.exists('vocabs/pretrained_vectors.pickle'):
            print('pretrained_vectors')
            with open('vocabs/pretrained_vectors.pickle', 'wb') as handle:
                pickle.dump(self.WORD.vocab.vectors, handle)
        print('finish vocab')

        # 4. make iterator for splits
        print('building iterators')
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        # 数据格式
        # id: '572a22256aef0514001552f8'
        # start_idx: 139
        # end_idx: 146
        # c_word: 二维数组, [[ 2, 1353, 188]], 每个数字代表一个单词，每个[]代表一个context
        # c_char: 三维数组, [[[3, 10, 2]]], 每个数字代表一个字母，[3, 10, 2]代表一个单词
        self.train_iter = data.BucketIterator(
            self.train,
            batch_size=args.train_batch_size,
            device=device,
            repeat=True,
            shuffle=True,
            sort_key=lambda x: len(x.c_word)
        )

        self.dev_iter = data.BucketIterator(
            self.dev,
            batch_size=args.dev_batch_size,
            device=device,
            repeat=False,
            sort_key=lambda x: len(x.c_word)
        )

    def preprocess(self, path):
        dump = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data['data']

        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                tokens = word_tokenize(context)
                abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

                for qas in paragraph['qas']:
                    id = qas['id']
                    question = qas['question']
                    for ans in qas['answers']:
                        answer = ans['text']
                        start_idx = ans['answer_start']
                        end_idx = start_idx + len(answer)
                        print('before', start_idx, end_idx)

                        # 原数据集中的start_idx, end_idx是字母级别的，和我们训练所需要的是单词级别(tokenized)的下标, 这里进行转换
                        char_index = 0
                        s_found = False
                        # TODO: 数据预处理没有全部做完
                        for i, t in enumerate(tokens):
                            while char_index < len(context):
                                if context[char_index] in abnormals:
                                    char_index += 1
                                else:
                                    break
                            # exceptional cases
                            if t[0] == '"' and context[char_index:char_index + 2] == '\'\'':
                                t = '\'\'' + t[1:]
                            elif t == '"' and context[char_index:char_index + 2] == '\'\'':
                                t = '\'\''

                            char_index += len(t)
                            if char_index > start_idx and s_found == False:
                                start_idx = i
                                s_found = True
                            if char_index > end_idx:
                                end_idx = i
                                break
                        tokens_len = len(tokens)
                        # 避免没有顺利找到index的情况，默认值设为len(tokens)-1
                        if start_idx >= tokens_len:
                            start_idx = tokens_len-1
                        if end_idx >= tokens_len:
                            end_idx = tokens_len-1
                        print('after', start_idx, end_idx)

                    dump.append(dict([
                        ('id', id),
                        ('context', context),
                        ('question', question),
                        ('answer', answer),
                        ('start_idx', start_idx),
                        ('end_idx', end_idx),
                    ]))

        with open('{}_processed'.format(path), 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="train-v1.1.json")
    parser.add_argument('--dev-file', default="dev-v1.1.json")
    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--train_batch_size', default=60, type=int)
    parser.add_argument('--dev_batch_size', default=100, type=int)
    parser.add_argument('--context_len', default=100, type=int)
    args = parser.parse_args()
    print('args', args)
    data = SQuAD(args)

