import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import os


# Dataset 클래스 정의
class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenized_text = tokenizer.encode(text)
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i : i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


# 학습 함수 정의
def train(model, tokenizer, train_dataset, device, epochs=3, batch_size=4, lr=5e-5):
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs
    )

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    dataset = ChatDataset(file_path="chat_data_temp.txt", tokenizer=tokenizer)
    model = train(model, tokenizer, dataset, device)

    # 모델 저장
    if not os.path.exists("model"):
        os.makedirs("model")
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    torch.save(model.state_dict(), "chatbot.pth")


if __name__ == "__main__":
    main()
