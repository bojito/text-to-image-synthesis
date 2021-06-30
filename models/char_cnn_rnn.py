import torch
import torch.nn as nn

from fixed_rnn import fixed_rnn

class fixed_rnn(nn.Module):
    '''
    Custom RNN that matches the implementation provided by Reed et al.

    We cannot use PyTorch's RNN because Reed's implementation does not use
    bias for the hidden layer (b_{hh}) during the computation of the first
    time step.
    '''
    def __init__(self, num_steps, emb_dim):
        super().__init__()
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)

        self.num_steps = num_steps
        self.relu = torch.nn.functional.relu


    def forward(self, txt):
        res = []
        for i in range(self.num_steps):
            i2h = self.i2h(txt[:, i]).unsqueeze(1)
            if i == 0:
                output = self.relu(i2h)
            else:
                h2h = self.h2h(res[i-1])
                output = self.relu(i2h + h2h)
            res.append(output)

        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res
        

class char_cnn_rnn(nn.Module):
    '''
    Char-CNN-RNN model, described in ``Learning Deep Representations of
    Fine-grained Visual Descriptions``.
    '''
    def __init__(self):
        super().__init__()

        rnn_dim = 512 
        
        rnn = fixed_rnn
        rnn_num_steps = 8
        
        # network setup
        # (B, 70, 201)
        self.conv1 = nn.Conv1d(70, 384, kernel_size=4)
        self.threshold1 = nn.Threshold(1e-6, 0)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        # (B, 384, 66)
        self.conv2 = nn.Conv1d(384, 512, kernel_size=4)
        self.threshold2 = nn.Threshold(1e-6, 0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        # (B, 512, 21)
        self.conv3 = nn.Conv1d(512, rnn_dim, kernel_size=4)
        self.threshold3 = nn.Threshold(1e-6, 0)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3,stride=2)
        
        # (B, rnn_dim, rnn_num_steps)
        self.rnn = rnn(num_steps=rnn_num_steps, emb_dim=rnn_dim)
        
        # (B, rnn_dim)
        self.emb_proj = nn.Linear(rnn_dim, 1024)
        # (B, 1024)


    def forward(self, txt):
        # temporal convolutions
        out = self.conv1(txt)
        out = self.threshold1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.threshold2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.threshold3(out)
        out = self.maxpool3(out)

        # recurrent computation
        out = out.permute(0, 2, 1)
        out = self.rnn(out)

        # linear projection
        out = self.emb_proj(out)

        return out
