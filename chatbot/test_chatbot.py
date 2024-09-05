import torch
import torch.nn as nn
import numpy as np
import os


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


def pad_sequences(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
    return padded_sequences


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    model = SimpleRNN(
        checkpoint["input_size"], checkpoint["hidden_size"], checkpoint["output_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, max_len


def predict_and_display_probabilities(
    model, vocab, input_text, max_len, max_output_words=5, temperature=1.0
):
    words = input_text.split()
    indices = [vocab.get(word, 0) for word in words]
    input_seq = pad_sequences([indices], max_len)
    input_tensor = torch.tensor(input_seq, dtype=torch.long)
    input_tensor = nn.functional.one_hot(
        input_tensor, num_classes=len(vocab) + 1
    ).float()

    output = model(input_tensor)
    output_probs = nn.functional.softmax(output / temperature, dim=2)

    output_seq = torch.multinomial(output_probs[0], 1).squeeze().tolist()

    inv_vocab = {v: k for k, v in vocab.items()}

    response_words = [inv_vocab.get(idx, "<UNK>") for idx in output_seq if idx != 0]
    response = " ".join(response_words[:max_output_words])

    print(f"ChatGSA: {response}\n")

    for i, word_idx in enumerate(output_seq[:max_output_words]):
        if word_idx == 0:
            continue
        word = inv_vocab.get(word_idx, "<UNK>")
        sorted_probs, indices = torch.sort(output_probs[0][i], descending=True)
        print(f"출력된 단어 '{word}' 뒤에 올 단어들의 확률:")
        for j, idx in enumerate(indices[:5]):
            print(
                f"  {inv_vocab.get(idx.item(), '<UNK>')}: {sorted_probs[j].item():.4f}"
            )


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "chatbot.pth")
    model, vocab, max_len = load_model(model_path)

    print("채팅이 시작되었습니다. '종료'를 입력하여 채팅을 종료할 수 있습니다.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "종료":
            break

        predict_and_display_probabilities(model, vocab, user_input, max_len)
