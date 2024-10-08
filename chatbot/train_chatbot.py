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


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out


def train_model(
    file_path,
    model_path,
    model_new_path,
    num_epochs=10,
    hidden_size=50,
    learning_rate=0.001,
):
    sequences, vocab = load_data(file_path)

    input_sequences = sequences[:-1]
    target_sequences = sequences[1:]

    max_len = max(len(seq) for seq in input_sequences + target_sequences)
    input_sequences = pad_sequences(input_sequences, max_len)
    target_sequences = pad_sequences(target_sequences, max_len)

    dataset = ChatDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    input_size = len(vocab) + 1
    output_size = len(vocab) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습에 사용할 연산 장치: {device}")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model = SimpleRNN(input_size, checkpoint["hidden_size"], output_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        max_len = checkpoint["max_len"]
        print("기존 모델이 발견되었습니다. 기존 모델에 이어서 학습시킵니다.")
    else:
        model = SimpleRNN(input_size, hidden_size, output_size)
        print("기존 모델이 발견되지 않아 새 모델을 생성 후 학습시킵니다.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        steps, step = len(dataloader), 0
        for inputs, targets in dataloader:
            inputs = nn.functional.one_hot(inputs, num_classes=input_size).float()
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1)

            outputs = model(inputs).view(-1, output_size)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 1000 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Step: {step}/{steps}")

        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": max_len,
            "hidden_size": hidden_size,
            "input_size": input_size,
            "output_size": output_size,
        },
        model_new_path,
    )
    print(f"모델이 {model_new_path}에 저장되었습니다.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, "chat_data.txt")
    model_path = os.path.join(current_dir, "chatbot.pth")
    model_new_path = os.path.join(current_dir, "chatbot.pth")

    num_epochs = 1
    hidden_size = 100
    learning_rate = 0.001

    train_model(
        file_path, model_path, model_new_path, num_epochs, hidden_size, learning_rate
    )
