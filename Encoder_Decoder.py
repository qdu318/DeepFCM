import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        out = self.dropout(src)
        outputs, (hidden, cell) = self.rnn(out)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(18, 32)
        self.out = nn.Linear(32, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]
        out = self.dropout(input)

        output, hidden = self.rnn(out, hidden)
        output = output.permute(0, 2, 1)
        prediction = self.out(output[:, :, -1])
        return prediction


class DeepPTP(nn.Module):
    def __init__(self, in_channels_S, out_channels_S, kernel_size_S, num_inputs_T, num_channels_T, num_outputs_T):
        super(DeepPTP, self).__init__()

        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=3)
        self.kernel_size = kernel_size_S

        self.linear = nn.Linear(64, num_outputs_T)

        self.encoder = Encoder(18, 32, 1, 0.2)
        self.decoder = Decoder(32, 64, 128, 0.2)
        self.fc = nn.Linear(2, 108)
        self.attr2atten = nn.Linear(1, 64)
        self.conv = nn.Conv1d(6, 32, 3)

    def forward(self, parameters):
        X = parameters['Start_Area']
        Y = parameters['End_Area']
        X, Y = torch.unsqueeze(X, dim=1), torch.unsqueeze(Y, dim=1)

        Time_Point = parameters['Time_Point']
        Prob = parameters['Prob']
        # Time_Point, Prob = torch.unsqueeze(torch.unsqueeze(Time_Point, dim=1), dim=2), torch.unsqueeze(torch.unsqueeze(Prob, dim=1), dim=2)
        Time_Point, Prob = torch.unsqueeze(Time_Point, dim=1), torch.unsqueeze(Prob, dim=1)

        # data_XY = torch.cat((X, Y, Time_Point, Prob), 1)

        data_XY = torch.cat((X, Y), 1)
        # data_XY = data_XY.float()
        # data_XY = torch.reshape(data_XY, (-1, 2))
        # data_XY = self.dropout2(data_XY)

        data_XY = self.fc(data_XY)  # [16,2]->[16,12]
        # data_XY = self.prelu(data_XY)
        data_XY = torch.reshape(data_XY, (-1, 6, 18))
        data_XY = self.conv(data_XY)
        data_XY = F.relu(data_XY)

        Time_Point, Prob = torch.unsqueeze(Time_Point.repeat(1, 32), dim=2), torch.unsqueeze(Prob.repeat(1, 32), dim=2)

        data_XY = torch.cat((data_XY, Time_Point, Prob), dim=2)

        locations = data_XY

        out1 = self.encoder(locations)
        out2 = self.decoder(locations, out1)
        # out2 = F.elu(out2)
        # lens = list(map(lambda x: x - self.kernel_size + 1, parameters['lens']))
        #
        # packed_inputs = nn.utils.rnn.pack_padded_sequence(locations, lens, batch_first=True)
        # packed_hiddens, (h_n, c_n) = self.lstm(packed_inputs)
        # hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)
        # hiddens = hiddens.permute(0, 2, 1)
        # out2 = self.linear(hiddens[:, :, -1])
        return F.log_softmax(out2, dim=1)
