import math
from einops import rearrange
import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.num_layers: int = num_layers
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size

        W_ii = [self.scaled_w((input_size, hidden_size), input_size)]
        W_if = [self.scaled_w((input_size, hidden_size), input_size)]
        W_ig = [self.scaled_w((input_size, hidden_size), input_size)]
        W_io = [self.scaled_w((input_size, hidden_size), input_size)]
        for i in range(num_layers - 1):
            W_ii.append(self.scaled_w((hidden_size, hidden_size), hidden_size))
            W_if.append(self.scaled_w((hidden_size, hidden_size), hidden_size))
            W_ig.append(self.scaled_w((hidden_size, hidden_size), hidden_size))
            W_io.append(self.scaled_w((hidden_size, hidden_size), hidden_size))
        W_hi = [self.scaled_w((hidden_size, hidden_size), hidden_size) for i in range(num_layers)]
        W_hf = [self.scaled_w((hidden_size, hidden_size), hidden_size) for i in range(num_layers)]
        W_hg = [self.scaled_w((hidden_size, hidden_size), hidden_size) for i in range(num_layers)]
        W_ho = [self.scaled_w((hidden_size, hidden_size), hidden_size) for i in range(num_layers)]
        
        b_i = [nn.Parameter(torch.zeros(hidden_size)) for i in range(num_layers)]
        b_f = [nn.Parameter(torch.ones(hidden_size)) for i in range(num_layers)]
        b_g = [nn.Parameter(torch.zeros(hidden_size)) for i in range(num_layers)]
        b_o = [nn.Parameter(torch.zeros(hidden_size)) for i in range(num_layers)]

        # output layer
        self.W_hy = self.scaled_w((hidden_size, input_size), hidden_size)
        self.b_y = nn.Parameter(torch.zeros(input_size))
    

        # register all parameters 
        self.W_ii = nn.ParameterList(W_ii)
        self.W_if = nn.ParameterList(W_if)
        self.W_ig = nn.ParameterList(W_ig)
        self.W_io = nn.ParameterList(W_io)
        self.W_hi = nn.ParameterList(W_hi)
        self.W_hf = nn.ParameterList(W_hf)
        self.W_hg = nn.ParameterList(W_hg)
        self.W_ho = nn.ParameterList(W_ho)
        self.b_i = nn.ParameterList(b_i)
        self.b_f = nn.ParameterList(b_f)
        self.b_g = nn.ParameterList(b_g)
        self.b_o = nn.ParameterList(b_o)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = torch.nn.functional.one_hot(x, num_classes=self.input_size).float()
        x = rearrange(x, "b s c -> s b c")

        h_t = torch.zeros(size=(self.num_layers, batch_size, self.hidden_size)) 
        c_t = torch.zeros(size=(self.num_layers, batch_size, self.hidden_size)) 
        h_t_minus_1 = h_t.clone()
        c_t_minus_1 = c_t.clone()



        output = []
        for t in range(seq_len):
            h_t = []
            c_t = []
            for i in range(self.num_layers):
                input_t = x[t] if i == 0 else h_t[i - 1]
                
                i_t = torch.sigmoid(
                    input_t @ self.W_ii[i] 
                    + h_t_minus_1[i] @ self.W_hi[i]
                    + self.b_i[i]
                )
                f_t = torch.sigmoid(
                    input_t @ self.W_if[i] 
                    + h_t_minus_1[i] @ self.W_hf[i]
                    + self.b_f[i]
                )
                g_t = torch.tanh(
                    input_t @ self.W_ig[i] 
                    + h_t_minus_1[i] @ self.W_hg[i]
                    + self.b_g[i]
                )
                o_t = torch.sigmoid(
                    input_t @ self.W_io[i] 
                    + h_t_minus_1[i] @ self.W_ho[i]
                    + self.b_o[i]
                )

                c_t.append(f_t * c_t_minus_1[i] + i_t * g_t)
                h_t.append(o_t * torch.tanh(c_t[i]))

            h_t = torch.stack(h_t)
            c_t = torch.stack(c_t)

            h_t_minus_1 = h_t
            c_t_minus_1 = c_t

            out = h_t[-1] @ self.W_hy + self.b_y 
            output.append(out)

        return torch.stack(output)


    def scaled_w(self, shape, fan_in):
            k = math.sqrt(1 / fan_in)
            W = (torch.rand(shape) - 0.5) * 2 * k
            return nn.Parameter(W)
