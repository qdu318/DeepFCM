import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPTP(nn.Module):

    def __init__(self, in_channels_S, out_channels_S, kernel_size_S, num_inputs_T, num_channels_T, num_outputs_T):
        super(DeepPTP, self).__init__()

        self.lstm = nn.LSTM(input_size=18, hidden_size=64, num_layers=1, batch_first=True)
        self.kernel_size = kernel_size_S
        self.linear = nn.Linear(64, num_outputs_T)
        self.prelu = torch.nn.PReLU(num_parameters=1,init=0.125)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(2, 108)
        self.attr2atten = nn.Linear(1, 64)
        self.conv = nn.Conv1d(6, 32, 3)

    def attent_pooling(self, hiddens, direction):
        direction = torch.unsqueeze(direction, dim=1).type(torch.float)
        attent = F.tanh(self.attr2atten(direction))
        attent = torch.unsqueeze(attent, dim=2)

        #hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)

        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim = 1, keepdim = True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)

        return hiddens

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

        data_XY = self.fc(data_XY) #[16,2]->[16,12]
        # data_XY = self.prelu(data_XY)
        data_XY = torch.reshape(data_XY, (-1, 6, 18))
        data_XY = self.conv(data_XY)
        data_XY = F.relu(data_XY)

        Time_Point, Prob = torch.unsqueeze(Time_Point.repeat(1, 32), dim=2), torch.unsqueeze(Prob.repeat(1, 32), dim=2)

        data_XY = torch.cat((data_XY, Time_Point, Prob), dim=2)


        locations = data_XY

        lens = list(map(lambda x : x - self.kernel_size + 1, parameters['lens']))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(locations, lens, batch_first=True)
        packed_hiddens, (h_n, c_n) = self.lstm(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        # hiddens = self.attent_pooling(hiddens, parameters['Time_Point']) # [16,64]
        hiddens = self.attent_pooling(hiddens, parameters['Start_Area'])
        # hiddens = hiddens.permute(0, 2, 1)
        # hiddens = torch.cat((hiddens1, hiddens2), dim=1) # [16,128]
        # hiddens = self.dropout2(hiddens)

        hiddens = F.elu(hiddens)

        out = self.linear(hiddens)
        aaa = F.log_softmax(out,dim=1)
        return aaa