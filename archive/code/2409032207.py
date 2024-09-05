import torch
import torch.nn as nn
from train_chatbot import ChatBotRNN, ChatDataset

MODEL_PATH = "chatbot.pth"
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


def load_model():
    dataset = ChatDataset("chat_data_temp.txt")
    model = ChatBotRNN(len(dataset.vocab) + 1, EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, dataset


def generate_response(model, dataset, input_sentence):
    words = input_sentence.split()
    input_seq = torch.tensor(
        [dataset.word2idx.get(word, 0) for word in words]
    ).unsqueeze(0)

    with torch.no_grad():
        output, _ = model(input_seq)

    output_idx = output.argmax(dim=-1).squeeze().tolist()
    response = " ".join([dataset.idx2word.get(idx, "<UNK>") for idx in output_idx])

    return response


if __name__ == "__main__":
    model, dataset = load_model()

    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = generate_response(model, dataset, user_input)
        print(f"Bot: {response}")
