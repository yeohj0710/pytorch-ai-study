import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import os

# 1. 텍스트 파일 읽기 및 전처리
file_path = "chat_data.txt"  # 채팅 데이터 파일 경로


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 각 줄을 단어로 분리
    texts = [line.strip().split() for line in lines]

    # 단어 빈도 계산 및 인덱스 매핑
    all_words = [word for text in texts for word in text]
    word_counts = Counter(all_words)
    vocab = {
        word: i + 1 for i, (word, _) in enumerate(word_counts.items())
    }  # 단어 -> 인덱스 매핑
    vocab["<PAD>"] = 0  # 패딩을 위한 인덱스 0 설정

    # 텍스트를 인덱스로 변환
    sequences = [[vocab[word] for word in text] for text in texts]

    return sequences, vocab


sequences, vocab = load_data(file_path)

# 입력/출력 쌍 생성
input_sequences = sequences[:-1]
target_sequences = sequences[1:]

# 시퀀스 패딩
max_len = max(len(seq) for seq in input_sequences + target_sequences)


def pad_sequences(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
    return padded_sequences


input_sequences = pad_sequences(input_sequences, max_len)
target_sequences = pad_sequences(target_sequences, max_len)


# 2. 데이터셋 및 DataLoader 정의
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


dataset = ChatDataset(input_sequences, target_sequences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 3. RNN 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out


input_size = len(vocab) + 1  # 단어의 개수 (vocab 크기) + 패딩을 위한 0
hidden_size = 50
output_size = len(vocab) + 1  # 출력도 단어의 개수와 동일

model = SimpleRNN(input_size, hidden_size, output_size)

# 4. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 인덱스 무시
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 모델 학습
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = nn.functional.one_hot(inputs, num_classes=input_size).float()
        targets = targets.view(-1)  # CrossEntropyLoss를 위해 shape 변경

        # Forward pass
        outputs = model(inputs).view(-1, output_size)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 6. 모델 저장
model_path = "simple_chatbot.pth"
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


# 7. 모델 불러오기 및 상호작용
def load_model(model_path):
    checkpoint = torch.load(model_path)
    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    model = SimpleRNN(
        checkpoint["input_size"], checkpoint["hidden_size"], checkpoint["output_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, max_len


def predict(model, vocab, input_text, max_len):
    words = input_text.split()
    indices = [vocab.get(word, 0) for word in words]
    input_seq = pad_sequences([indices], max_len)
    input_tensor = torch.tensor(input_seq, dtype=torch.long)
    input_tensor = nn.functional.one_hot(
        input_tensor, num_classes=len(vocab) + 1
    ).float()

    output = model(input_tensor)
    output_seq = torch.argmax(output, dim=2).squeeze().tolist()

    inv_vocab = {v: k for k, v in vocab.items()}
    response = " ".join(inv_vocab.get(idx, "<UNK>") for idx in output_seq if idx != 0)

    return response


model, vocab, max_len = load_model(model_path)

print("챗봇이 준비되었습니다. 'exit'를 입력하여 종료하세요.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = predict(model, vocab, user_input, max_len)
    print(f"Bot: {response}")
