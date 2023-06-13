import re
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

chars_sorted_tadjanovic = ['\n', ' ', '!', '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                           'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z',
                           '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                           'r', 's', 't', 'u', 'v', 'x', 'z', '«', '»', 'Ú', 'à', 'á', 'â', 'è', 'é', 'ê', 'ì', 'í',
                           'î', 'ò', 'ó', 'ô', 'ú', 'ü', 'ā', 'Ć', 'ć', 'Č', 'č', 'Đ', 'đ', 'ē', 'ī', 'ŕ', 'Š', 'š',
                           'Ž', 'ž', 'ȁ', 'ȅ', 'ȉ', 'ȍ', 'ȕ', '—']
chars_sorted_fcm = ['\n', ' ', '!', "'", '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B',
                    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
                    'X', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'Ć', 'ć', 'Č', 'č', 'Đ', 'đ', 'Š', 'š', 'Ž', 'ž']


def sample(model, starting_str,
           len_generated_text=500,
           scale_factor=1.0):
    model_name = model
    if model == "tadjanovic":
        model = model_tadjanovic
        char2int = char2int_tadjanovic
        char_array = char_array_tadjanovic
    elif model == "fcm":
        model = model_fcm
        char2int = char2int_fcm
        char_array = char_array_fcm
    else:
        raise RuntimeError("invalid model!")

    encoded_input = torch.tensor(
        [char2int[s] for s in starting_str]
    )
    encoded_input = torch.reshape(
        encoded_input, (1, -1)
    )
    generated_str = starting_str

    model.eval()

    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model(
            encoded_input[:, c].view(1), hidden, cell
        )

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(
            last_char.view(1), hidden, cell
        )
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])
    generated_str = re.sub(" +", " ", re.sub("\n+", "\n", generated_str))
    if model_name == "tadjanovic":
        if "[EOS]" in generated_str:
            return generated_str.split("[EOS]")[0]
    for i in range(-1, -len(generated_str), -1):
        if generated_str[i] == ".":
            generated_str = generated_str[0:i]
            return generated_str + "."

    return generated_str


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size,
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).un_squeeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell


char_array_tadjanovic = np.array(chars_sorted_tadjanovic)
char2int_tadjanovic = {ch: i for i, ch in enumerate(chars_sorted_tadjanovic)}
char_array_fcm = np.array(chars_sorted_fcm)
char2int_fcm = {ch: i for i, ch in enumerate(chars_sorted_fcm)}
model_tadjanovic = torch.load("TADJANOVIC_LSTM_MODEL.pt")
model_tadjanovic.eval()
model_fcm = torch.load("FCM_LSTM_MODEL.pt")
model_fcm.eval()

text = sample("fcm","Živlenje o kak si mi rad!",1000,1)
print(text)