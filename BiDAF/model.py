import torch
import torch.nn as nn
import torch.nn.functional as F
from BiDAF.utils.nn import Linear, LSTM


class BiDAF(nn.Module):
    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        # padding_idx: 对较短的单词进行padding(填充）时，填充的是1
        # TODO: 所需要的数据格式？
        self.char_emb = nn.Embedding(self.args.char_vocab_size, self.args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, self.args.char_channel_size, (self.args.char_dim, self.args.char_channel_width)),
            nn.ReLU()
        )

        # 1. Word Embedding Layer
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # 2. highway network TODO:
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(self.args.hidden_size * 2, self.args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(self.args.hidden_size * 2, self.args.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=self.args.hidden_size * 2,
                                 hidden_size=self.args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.args.dropout_rate)

        # 4. Attention Flow Layer
        self.weight_c_att = Linear(self.args.hidden_size*2, 1)
        self.weight_q_att = Linear(self.args.hidden_size*2, 1)
        self.weight_cq_att = Linear(self.args.hidden_size*2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=self.args.hidden_size * 8,
                                   hidden_size=self.args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.args.dropout_rate)

        self.modeling_LSTM2 = LSTM(input_size=self.args.hidden_size * 2,
                                   hidden_size=self.args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.args.dropout_rate)

        # 6. Output Layer
        self.pStart_weight_g = Linear(self.args.hidden_size * 8, 1, dropout=self.args.dropout_rate)
        self.pStart_weight_m = Linear(self.args.hidden_size * 2, 1, dropout=self.args.dropout_rate)
        self.pEnd_weight_g = Linear(self.args.hidden_size * 8, 1, dropout=self.args.dropout_rate)
        self.pEnd_weight_m = Linear(self.args.hidden_size * 2, 1, dropout=self.args.dropout_rate)

        self.output_LSTM = LSTM(
                                input_size=self.args.hidden_size * 2,
                                hidden_size=self.args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.args.dropout_rate)
        self.dropout = nn.Dropout(p=self.args.dropout_rate)

    def forward(self, batch):
        def char_emb_layer(x):
            """
            :param x: (batch_size, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch_size, seq_len, word_len, char_dim)
            x = self.char_emb(x)
            # (batch_size, seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch_size * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            # (batch_size * seq_len, char_channel_size, 1, conv_len) ->
            # (batch_size * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze(2)
            # (batch_size * seq_len, char_channel_size, 1) ->
            # (batch_size * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch_size, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # 对Character Embedding 和 Word Embedding 的结果进行拼接
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch_size, c_len, hidden_size * 2)
            :param q: (batch_size, q_len, hidden_size * 2)
            :return: (batch_size, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                # (batch_size, 1, hidden_size*2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch_size. c_len, 1)
                ci = self.weight_cq_att(c * qi).squeeze()
                cq.append(ci)

            # (batch_size, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch_size, c_len, q_len)
            s = self.weight_c_att(c).expand(-1, -1, q_len) + self.weight_q_att(q).permute(0, 2, 1).expand(-1, c_len, -1) + cq

            # (batch_size, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch_size, c_len, q_len) * (batch_size, q_len, hidden_size * 2) -> (batch_size, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch_size, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch_size, 1, c_len) * (batch_size, c_len, hidden_size * 2) -> (batch_size, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch_size, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)

            # (batch_size, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch_size, c_len, hidden_size * 8)
            :param m: (batch_size, c_len ,hidden_size * 2)
            :return: p1: (batch_size, c_len), p2: (batch_size, c_len)
            """
            # (batch_size, c_len)
            pStart = (self.pStart_weight_g(g) + self.pStart_weight_m(m)).squeeze()
            # (batch_size, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch_size, c_len)
            pEnd = (self.pEnd_weight_g(g) + self.pEnd_weight_m(m2)).squeeze()

            return pStart, pEnd

        # 1. Character Embedding Layer
        # before embedding: c_char [batch_size, 49, 22] q_char [batch_size, 30, 16]
        # c_char: [100, 324, 100] [batch_size, seq_len, word_dim]
        # q_char: [100, 24, 100]

        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)

        # 2. Word Embedding Layer
        # before embedding:
        # c_word[0]: [batch_size, 49]
        # c_word[1]: [batch_size]
        # q_word[0]: [batch_size, 30]
        # q_word[1]: [batch_size]

        # c_word: [100, 324, 100]
        # q_word: [100, 24, 100]
        # c_lens: [100] [139,186,65....]
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        # c [100, 236, 200] (batch_size, seq_len, hidden_size * 2)
        # q [100, 23, 200]
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)

        # 3. Contextual Embedding Layer TODO:
        # c [100, 303, 200] (batch_size, , 2*hidden_size)
        # q [100, 26, 200]
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]

        # 4. Attention Flow Layer
        # g [100, 303, 800] (batch_size, , 8*hidden_size)
        g = att_flow_layer(c, q)

        # 5. Modeling Layer
        # m [100, 303, 200] (batch_size, , 2*hidden_size)
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]

        # 6. Output Layer
        pStart, pEnd = output_layer(g, m, c_lens)

        return pStart, pEnd



