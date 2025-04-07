import torch
import torch.nn as nn
from torch.autograd import Variable


class BaseModel(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum

        if cell == "RNN":
            self.cell = nn.RNN(
                input_size=self.inputDim,
                hidden_size=self.hiddenNum,
                num_layers=self.layerNum,
                dropout=0.0,
                nonlinearity="tanh",
                batch_first=True,
            )
        if cell == "LSTM":
            self.cell = nn.LSTM(
                input_size=self.inputDim,
                hidden_size=self.hiddenNum,
                num_layers=self.layerNum,
                dropout=0.0,
                batch_first=True,
            )
        if cell == "GRU":
            self.cell = nn.GRU(
                input_size=self.inputDim,
                hidden_size=self.hiddenNum,
                num_layers=self.layerNum,
                dropout=0.0,
                batch_first=True,
            )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


class LSTMModel(BaseModel):
    def __init__(self, device="cpu", args: dict = None):
        self.device = device
        inputDim = args["input_dim"]
        hiddenNum = args["hidden_dim"]
        outputDim = args["output_dim"]
        layerNum = args["layer_num"]
        cell = args["cell"]
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    def forward(self, x):
        batchSize = x.size(0)
        # print(batchSize)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        rnnOutput, hn = self.cell(x, (h0, c0))

        if self.layerNum > 1:
            hn = hn[0][-1].view(batchSize, self.hiddenNum)
        else:
            hn = hn[0].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput
