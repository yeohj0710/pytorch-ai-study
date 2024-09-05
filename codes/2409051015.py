import torch
from train_chatbot import GPTPretrain, Config


def load_model(model_path, config):
    model = GPTPretrain(config)
    model.load(model_path)
    model.to(config.device)
    return model


def chat_with_bot(model, vocab, user_input):
    model.eval()
    with torch.no_grad():
        tokens = vocab.encode_as_pieces(user_input)
        input_ids = (
            torch.tensor([vocab.piece_to_id(token) for token in tokens])
            .unsqueeze(0)
            .to(model.config.device)
        )
        output, _ = model(input_ids)
        output_ids = torch.argmax(output, dim=-1).squeeze().tolist()
        response = vocab.decode_pieces([vocab.id_to_piece(id) for id in output_ids])
        return response


def main():
    # Config 설정
    config = Config(
        {
            "n_dec_vocab": 8007,
            "n_dec_seq": 256,
            "n_layer": 6,
            "d_hidn": 256,
            "i_pad": 0,
            "d_ff": 1024,
            "n_head": 4,
            "d_head": 64,
            "dropout": 0.1,
            "layer_norm_epsilon": 1e-12,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
    )

    # Vocab 및 모델 로드
    vocab = load_vocab()  # 사용할 SentencePiece 모델 로드
    model = load_model("chatbot_model.pth", config)

    # 챗봇 대화
    print("Start chatting with the bot! (Type 'quit' to stop)")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        response = chat_with_bot(model, vocab, user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
