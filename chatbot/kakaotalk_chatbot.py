import pyautogui
import time
import pyperclip
import torch
import torch.nn as nn
import numpy as np
import os


def get_text_from_area(x1, y1, x2, y2):
    pyautogui.moveTo(x1, y1)
    pyautogui.mouseDown()
    pyautogui.moveTo(x2, y2)
    pyautogui.mouseUp()
    pyautogui.hotkey("ctrl", "c")
    time.sleep(1)
    return pyperclip.paste()


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


def predict_random(
    model, vocab, input_text, max_len, max_output_words=5, temperature=1.0
):
    """확률을 반영하여 단어를 선택하되, 무작위성을 부여"""
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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    end_char_file_path = os.path.join(current_dir, "end_char.txt")

    with open(end_char_file_path, "r", encoding="utf-8") as f:
        end_chars = [line.strip() for line in f if line.strip()]

    response = ""
    for word in response_words:
        response += word + " "
        if any(word.endswith(end_char) for end_char in end_chars):
            break

    response_words = response.strip().split()[:max_output_words]
    response = " ".join(response_words)

    return response


def main():
    x1, y1 = 1456, 825
    x2, y2 = 1898, 866
    prev_text = ""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "chatbot.pth")
    model, vocab, max_len = load_model(model_path)

    while True:
        time.sleep(3)
        current_text = get_text_from_area(x1, y1, x2, y2)

        if current_text == prev_text:
            continue

        if "깡통:" in current_text:
            continue

        if "깡통" not in current_text:
            continue

        print(f"New Text: {current_text}")

        filtered_text = current_text.replace("깡통", "").strip()
        if not filtered_text:
            prev_text = current_text
            continue

        response = predict_random(model, vocab, filtered_text, max_len)
        response_text = f"깡통: {response}"
        print(response_text)

        pyautogui.click(1500, 950)
        time.sleep(0.5)

        pyperclip.copy(response_text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)

        pyautogui.press("enter")
        time.sleep(0.5)

        prev_text = response_text


if __name__ == "__main__":
    main()
