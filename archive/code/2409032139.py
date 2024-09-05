import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("model")
    model = GPT2LMHeadModel.from_pretrained("model")
    model.load_state_dict(torch.load("chatbot.pth"))
    model.to(device)
    model.eval()

    # pad_token_id를 eos_token_id로 설정
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        input_text = input("You: ")
        if input_text.lower() == "quit":
            break

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {output_text}")


if __name__ == "__main__":
    main()
