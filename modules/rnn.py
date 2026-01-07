from einops.einops import rearrange
import torch
import torch.nn as nn
import math

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers=1, nonlinearity='tanh', bias=True, device=None, dtype=None):
        super().__init__()
        # note: maybe also transpose twice for efficiency?
        scale_in: float = math.sqrt(1 / input_size)
        scale_h: float = math.sqrt(1 / hidden_size)
        self.num_layers: int = num_layers
        self.W_ih = nn.Parameter(torch.rand((input_size, hidden_size), device=device, dtype=dtype) * scale_in * 2 - scale_in)
        self.W_hh = nn.Parameter(torch.rand((num_layers, hidden_size, hidden_size), device=device, dtype=dtype) * scale_h * 2 - scale_h)
        self.W_h_h = nn.Parameter(torch.rand((num_layers - 1, hidden_size, hidden_size), device=device, dtype=dtype) * scale_h * 2 - scale_h)
        self.b_h = nn.Parameter(torch.zeros(num_layers, hidden_size, device=device, dtype=dtype))
        self.W_hy = nn.Parameter(torch.rand((hidden_size, input_size), device=device, dtype=dtype) * scale_h * 2 - scale_h)
        self.b_y = nn.Parameter(torch.zeros(input_size, device=device, dtype=dtype))

        self.hidden_size: int = hidden_size

    def forward(self, x, hx=None):
        batch_size, seq_len = x.shape
        x = torch.nn.functional.one_hot(x, num_classes=self.W_ih.shape[0]).float()
        x = rearrange(x, "b s c -> s b c")
         
        if not hx:
           hx = torch.zeros((self.num_layers, batch_size, self.hidden_size)) 

        h_t_minus_1 = hx.clone()

        output = []
        for t in range(seq_len):
            h_t = []
            for i in range(self.num_layers):
                if i == 0:
                    input_t = x[t]
                    h_t.append(torch.tanh(
                        input_t @ self.W_ih 
                        + h_t_minus_1[i] @ self.W_hh[i]
                        + self.b_h[i]
                    ))
                else:
                    input_t = h_t[i - 1]
                    h_t.append(torch.tanh(
                        input_t @ self.W_h_h[i - 1] 
                        + h_t_minus_1[i] @ self.W_hh[i]
                        + self.b_h[i]
                    ))
 
            h_t = torch.stack(h_t)
            out = h_t[-1] @ self.W_hy + self.b_y 
            output.append(out)
            h_t_minus_1 = h_t.clone()

        output = torch.stack(output)
        return output

    def sample(self, prefix, length, temperature=1.0):
        # first prefill with prefix
        seq_len = len(prefix)
        x = torch.nn.functional.one_hot(prefix, num_classes=self.W_ih.shape[0]).float()
 
        h_t_minus_1 = torch.zeros((self.num_layers, self.hidden_size))
        h_t = torch.zeros((self.num_layers, self.hidden_size))

        output = []
        for t in range(seq_len):
            h_t = []
            for i in range(self.num_layers):
                if i == 0:
                    input_t = x[t]
                    h_t.append(torch.tanh(
                        input_t @ self.W_ih 
                        + h_t_minus_1[i] @ self.W_hh[i]
                        + self.b_h[i]
                    ))
                else:
                    input_t = h_t[i - 1]
                    h_t.append(torch.tanh(
                        input_t @ self.W_h_h[i - 1] 
                        + h_t_minus_1[i] @ self.W_hh[i]
                        + self.b_h[i]
                    ))
 
            h_t = torch.stack(h_t)
            out = h_t[-1] @ self.W_hy + self.b_y 
            print(out)
            output.append(out)
            h_t_minus_1 = h_t.clone()

        output = prefix
        # now sample
        for t in range(length - seq_len):
            logits = out / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            output = torch.cat((output, next_token), dim=0)

            input_t = torch.nn.functional.one_hot(next_token.squeeze(), num_classes=self.W_ih.shape[0]).float()
            h_t = []
            for i in range(self.num_layers):
                if i == 0:
                    input_t = input_t
                    h_t.append(torch.tanh(
                        input_t @ self.W_ih 
                        + h_t_minus_1[i] @ self.W_hh[i]
                        + self.b_h[i]
                    ))
                else:
                    input_t = h_t[i - 1]
                    h_t.append(torch.tanh(
                        input_t @ self.W_h_h[i - 1] 
                        + h_t_minus_1[i] @ self.W_hh[i]
                        + self.b_h[i]
                    ))
 
            h_t = torch.stack(h_t)
            out = h_t[-1] @ self.W_hy + self.b_y 
            
        return output

