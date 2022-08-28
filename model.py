import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class PHVModel(nn.Module):
    def __init__(self, max_word, word_embedding_size=32, lstm_hidden_size=32, lstm_layers=3, hidden_size=32):
        super(PHVModel, self).__init__()

        self.embedding_layer = nn.Embedding(max_word, word_embedding_size)
        self.lstm_layer = nn.LSTM(word_embedding_size, lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=lstm_layers)
        self.linear_layer = nn.Linear(lstm_hidden_size * 2, hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, seq1, seq2, len1, len2):
        seq1_embedding = self.embedding_layer(seq1)
        seq2_embedding = self.embedding_layer(seq2)

        seq1_embedding = rnn.pack_padded_sequence(seq1_embedding, len1, batch_first=True, enforce_sorted=False)
        seq2_embedding = rnn.pack_padded_sequence(seq2_embedding, len2, batch_first=True, enforce_sorted=False)

        seq1_embedding, _ = self.lstm_layer(seq1_embedding)
        seq2_embedding, _ = self.lstm_layer(seq2_embedding)

        seq1_lstm_out, lens_unpacked1 = rnn.pad_packed_sequence(seq1_embedding, batch_first=True)
        seq2_lstm_out, lens_unpacked2 = rnn.pad_packed_sequence(seq2_embedding, batch_first=True)

        seq1_lstm_out = torch.cat([seq1_lstm_out[i:i+1, lens_unpacked1[i]-1, :] for i in range(lens_unpacked1.size()[0])], dim=0)
        seq2_lstm_out = torch.cat([seq2_lstm_out[i:i+1, lens_unpacked2[i]-1, :] for i in range(lens_unpacked2.size()[0])], dim=0)

        seq1_lstm_out = seq1_lstm_out.view(seq1_lstm_out.size()[0], -1)
        seq2_lstm_out = seq2_lstm_out.view(seq2_lstm_out.size()[0], -1)

        seq1_embedded = self.linear_layer(seq1_lstm_out)
        seq2_embedded = self.linear_layer(seq2_lstm_out)

        out = torch.cat([seq1_embedded, seq2_embedded], dim=1)
        out = self.output_layer(out)
        return out


# from torch.utils.data import Dataset
# seq1 = [torch.tensor([1, 2, 3, 4, 5, 6, 7]),
#            torch.tensor([2, 3, 4, 5, 6, 7]),
#            torch.tensor([3, 4, 5, 6, 7]),
#            torch.tensor([4, 5, 6, 7]),
#            torch.tensor([5, 6, 7]),
#            torch.tensor([6, 7]),
#            torch.tensor([7])]

# class MyData(Dataset):
#     def __init__(self, seq1, seq2):
#         self.seq1 = seq1
#         self.seq2 = seq2

#     def __len__(self):
#         return len(self.seq1)

#     def __getitem__(self, item):
#         return self.seq1[item], self.seq2[item]

# def collate_fn(train_data):
#     train_data.sort(key=lambda data: len(data), reverse=True)
#     data_length1 = [len(data[0]) for data in train_data]
#     data_length2 = [len(data[1]) for data in train_data]
#     seq1 = []
#     seq2 = []
#     for data in train_data:
#         seq1.append(data[0])
#         seq2.append(data[1])

#     seq1 = rnn.pad_sequence(seq1, batch_first=True)
#     seq2 = rnn.pad_sequence(seq2, batch_first=True)
#     return seq1, seq2, torch.tensor(data_length1), torch.tensor(data_length2)

# train_data = MyData(seq1, seq1)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)

# model = PHVModel(max_word=10)

# for data in train_loader:
#     seq1, seq2, len1, len2 = data
#     print(seq1.size())
#     print(seq2.size())
#     print(len1.size())
#     print(len2.size())
#     print(model(seq1, seq2, len1, len2))
#     break





