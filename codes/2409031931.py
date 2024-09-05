import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import os


def load_data(file_path, sequence_length=5):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    texts = [line.strip() for line in lines if line.strip()]
    sequences = []
    for i in range(len(texts) - sequence_length):
        sequence = texts[i : i + sequence_length]
        sequences.append(sequence)

    all_words = [
        word for sequence in sequences for text in sequence for word in text.split()
    ]
    word_counts = Counter(all_words)
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())}
    vocab["<PAD>"] = 0

    sequences = [
        [vocab[word] for text in sequence for word in text.split()]
        for sequence in sequences
    ]

    return sequences, vocab


def pad_sequences(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
    return padded_sequences


class ChatDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.input_sequences[idx], dtype=torch.long), torch.tensor(
            self.target_sequences[idx], dtype=torch.long
        )


class ImprovedRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=25):
        super(ImprovedRNN, self).__init__()  # 올바르게 부모 클래스 초기화
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out


def train_model(
    file_path, model_path, num_epochs=10, hidden_size=50, learning_rate=0.001
):
    sequences, vocab = load_data(file_path)

    input_sequences = sequences[:-1]
    target_sequences = sequences[1:]

    max_len = max(len(seq) for seq in input_sequences + target_sequences)
    input_sequences = pad_sequences(input_sequences, max_len)
    target_sequences = pad_sequences(target_sequences, max_len)

    dataset = ChatDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_size = len(vocab) + 1
    output_size = len(vocab) + 1

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model = ImprovedRNN(input_size, checkpoint["hidden_size"], output_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        max_len = checkpoint["max_len"]
        print("기존 모델 불러옴.")
    else:
        model = ImprovedRNN(input_size, output_size, hidden_size)
        print("새 모델 생성.")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = nn.functional.one_hot(inputs, num_classes=input_size).float()
            targets = targets.view(-1)

            outputs = model(inputs).view(-1, output_size)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": max_len,
            "hidden_size": hidden_size,
            "input_size": input_size,
            "output_size": output_size,
        },
        model_path,
    )
    print("모델 저장 완료:", model_path)


if __name__ == "__main__":
    file_path = "chat_data.txt"
    model_path = "simple_chatbot.pth"
    num_epochs = 30
    hidden_size = 50
    learning_rate = 0.001

    train_model(file_path, model_path, num_epochs, hidden_size, learning_rate)
