import torch

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, args, embedding):
        super().__init__()
        # self.input_size = 5 + embedding
        self.input_size = embedding + 1
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.num_directions = 1
        self.batch_size = args.batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h_t, c_t) = self.lstm(input_seq, (h_0, c_0))

        pred = self.linear(output)  # pred(batch_size, seq_len, output_size)
        pred = pred[:, -1, :]
        return output, pred, h_t, c_t


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.num_directions = 1
        self.batch_size = args.batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dis = nn.Softplus()

    def forward(self, input_seq, h_t, c_t):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        output, (h_t, c_t) = self.lstm(input_seq, (h_t, c_t))
        pred = output[:, -1, :]
        pred = self.dis(pred)
        return pred, h_t, c_t