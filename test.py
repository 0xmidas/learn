from einops.einops import rearrange
from tqdm import tqdm
from modules.rnn import RNN 
import torch

text_file = "./data/shakespeare.txt"

with open(text_file, "r") as f:
    text = f.read()
print(f"Length of dataset in characters: {len(text)}")

STOP_TOKEN = 27
VOCAB_SIZE = 28  # 26 letters + space + stop
HIDDEN_SIZE = 512 
BATCH_SIZE = 16 
CHUNK_SIZE = 256


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


model = RNN(VOCAB_SIZE, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(2048):
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

        logits = model(inputs)

        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 32 == 0:
        print(loss.item())



test = "i"
output = model.sample(torch.tensor(text_to_tokens(test, stop=False)), 100)
print(tokens_to_text(output))

