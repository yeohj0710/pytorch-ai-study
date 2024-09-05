import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# 하이퍼파라미터 설정
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "chatbot.pth"


# 데이터셋 클래스 정의
class ChatDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

        self.lines = [line.strip() for line in self.lines if len(line.strip()) > 0]
        self.vocab = set(word for line in self.lines for word in line.split())
        self.word2idx = {word: idx + 1 for idx, word in enumerate(self.vocab)}
        self.word2idx["<PAD>"] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.data = [
            (self.preprocess(line), self.preprocess(self.lines[i + 1]))
            for i, line in enumerate(self.lines[:-1])
        ]

    def preprocess(self, line):
        return [self.word2idx[word] for word in line.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )
    return inputs, targets


# RNN 모델 정의
class ChatBotRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ChatBotRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden


def train_model():
    dataset = ChatDataset("chat_data_temp.txt")
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = ChatBotRNN(len(dataset.vocab) + 1, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)

            batch_size, seq_len, vocab_size = outputs.size()
            outputs = outputs.view(-1, vocab_size)

            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
