import torch
import torch.nn as nn
import numpy as np


class ImprovedRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(ImprovedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out


def pad_sequences(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
    return padded_sequences


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    model = ImprovedRNN(
        checkpoint["input_size"], checkpoint["output_size"], checkpoint["hidden_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, max_len


def predict_with_sampling(
    model,
    vocab,
    input_text,
    max_len,
    temperature=0.8,
    top_k=10,
    top_p=0.9,
    repetition_penalty=1.2,
):
    words = input_text.split()
    indices = [vocab.get(word, 0) for word in words]
    input_seq = pad_sequences([indices], max_len)
    input_tensor = torch.tensor(input_seq, dtype=torch.long)
    input_tensor = nn.functional.one_hot(
        input_tensor, num_classes=len(vocab) + 1
    ).float()

    output = model(input_tensor)
    output = output.squeeze(0) / temperature

    sorted_probs, sorted_indices = torch.sort(output, descending=True)
    cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_probs, dim=-1), dim=-1
    )

    top_p_mask = cumulative_probs > top_p
    top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
    top_p_mask[:, 0] = 0

    sorted_probs[top_p_mask] = float("-inf")

    top_k_probs, top_k_indices = torch.topk(sorted_probs, top_k, dim=-1)
    top_k_probs = torch.nn.functional.softmax(top_k_probs, dim=-1)
    sampled_indices = torch.multinomial(top_k_probs, 1).squeeze().tolist()

    inv_vocab = {v: k for k, v in vocab.items()}
    response = " ".join(
        inv_vocab.get(idx, "<UNK>") for idx in sampled_indices if idx != 0
    )

    words = response.split()
    filtered_words = []
    last_word = None
    repeat_count = 0

    for word in words:
        if word == last_word:
            repeat_count += 1
        else:
            repeat_count = 0

        if repeat_count < 3:
            filtered_words.append(word)

        last_word = word

    return " ".join(filtered_words)


if __name__ == "__main__":
    model_path = "simple_chatbot.pth"
    model, vocab, max_len = load_model(model_path)

    print("챗봇이 준비되었습니다. 'exit'를 입력하여 종료하세요.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = predict_with_sampling(model, vocab, user_input, max_len)
        print(f"Bot: {response}")
