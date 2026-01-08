import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from einops.einops import rearrange
from tqdm import tqdm
from modules.lstm import LSTM
import torch

text_file = "./data/shakespeare.txt"

with open(text_file, "r") as f:
   text = f.read()

print(f"Length of dataset in characters: {len(text)}")


STOP_TOKEN = 27
VOCAB_SIZE = 28  # 26 letters + space + stop
HIDDEN_SIZE = 64
BATCH_SIZE = 128 
CHUNK_SIZE = 64
NUM_LAYERS = 3


def text_to_tokens(text, stop=False):
    """
    0-25: a-z
    26: space
    27: stop
    """
    tokens = []
    for char in text.lower():
        if 'a' <= char <= 'z':
            tokens.append(ord(char) - ord('a'))
        elif char == ' ':
            tokens.append(26)
        # else: skip unknown characters
    if stop:
        tokens.append(27)  # add stop token
    return tokens

data = torch.tensor(text_to_tokens(text))

def tokens_to_text(tokens):
    """
    0-25: a-z
    26: space
    27: stop (ends conversion)
    """
    chars = []
    for token in tokens:
        if 0 <= token and token <= 25:
            chars.append(chr(token + ord('a')))
        elif token == 26:
            chars.append(' ')
        elif token == 27:
            break  # stop token
    return ''.join(chars)


model = LSTM(VOCAB_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(16 * 32):
    for i in range(0, len(data) - CHUNK_SIZE - 1, CHUNK_SIZE * BATCH_SIZE):
        inputs = []
        targets = []
        for b in range(BATCH_SIZE):
            start = i + b * CHUNK_SIZE
            if start + CHUNK_SIZE + 1 > len(data):
                break
            inputs.append(data[start : start + CHUNK_SIZE])
            targets.append(data[start + 1: start + CHUNK_SIZE + 1])

        if len(inputs) < BATCH_SIZE:
            continue

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        logits = model(inputs)  # (seq_len, batch, vocab)
        logits = logits.transpose(0, 1)  # (batch, seq_len, vocab)

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.view(-1))


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        
    if epoch % 32 == 0:
        print(loss.item())



