# 加性注意力
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from model.encoder_decoder import Encoder, Decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, key, value, choose):
        """
        Args:
            query: (N, n, d_q)
            key: (N, m, d_k)
            value: (N, m, d_v)
        """
        if choose == 0:
            query, key = self.W_q(query), self.W_k(key)
            features = query.unsqueeze(2) + key.unsqueeze(1)
            features = torch.tanh(features)
            scores = self.W_v(features).squeeze(-1)
            attn_weights = F.softmax(scores, dim=1)
        else:
            attn_weights = F.softmax(torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.size(2)), dim=-1)
        return torch.bmm(attn_weights, value)


class Model(nn.Module):
    """
    定义一个自动编码器的子类，继承父类 nn.Module
    并且自动编码器通过编码器和解码器传递输入
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.Attention_hidden_size = args.Attention_hidden_size
        self.CNN_hid1 = args.CNN_hid1
        self.kernel1 = args.kernel1
        self.k = args.long_interval // args.short_interval
        self.feature_dim = self.k + args.cov
        self.Encoder_hidden_size = args.hidden_size
        self.embedding = round(math.sqrt((self.feature_dim - 2 * (self.kernel1 - 1))))
        # self.embedding = round(math.sqrt((self.k - 1 * (self.kernel1 - 1))))
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.input_size, self.input_size, self.kernel1),
            nn.Softplus(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.input_size, self.input_size, self.kernel1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Linear((self.feature_dim - 2 * (self.kernel1 - 1)), self.embedding),
        )
        self.encoder = Encoder(args, self.embedding).to(device)
        self.decoder = Decoder(args).to(device)

        self.linear = nn.Linear(self.Encoder_hidden_size, 1)
        self.dis = nn.Softplus()

    def forward(self, x):
        # x:(batch, seq_length, feature)
        # Encoder
        batch = x.shape[0]
        outputs = torch.zeros(batch, self.output_size).to(device)

        dense_series = x[:, :, :self.feature_dim]  # 提取密集序列
        layer1_output = self.layer1(dense_series)  # 对密集序列进行卷积
        layer2_output = self.layer2(layer1_output)  # 对密集序列进行卷积
        linear_output = self.layer3(layer2_output)  # 线性层
        lstm_input = torch.cat([linear_output, x[:, :, self.feature_dim:]], dim=2)

        encoder_output, pre,  h_t, c_t = self.encoder(lstm_input)  # encoder_output(batch, train_window, hidden)
        s_t = encoder_output[:, -1, :]  # 训练窗口预测的下一时刻的值
        encoder_output = encoder_output[:, :-1, :]
        # Decoder

        for t in range(0, self.output_size):

            att = AdditiveAttention(self.Encoder_hidden_size, self.Encoder_hidden_size, self.Attention_hidden_size).to(device)  # 加性注意力机制
            if t >= 0:
                encoder_output = torch.cat([encoder_output, s_t.unsqueeze(1)], dim=1)
            decoder_input = att(torch.cat([h_t.permute(1, 0, 2), c_t.permute(1, 0, 2)], dim=1), encoder_output,
                                encoder_output, 0)  # decoder_input (batch, 1, attention_hidden)

            s_t, h_t, c_t = self.decoder(decoder_input, h_t, c_t)  # h_t (1, batch, attention_hidden)
            outputs[:, t] = self.linear(s_t).squeeze()

        return outputs