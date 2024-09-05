import json
from tqdm import tqdm


def tokenize_text(text):
    """
    간단한 토큰화 함수.
    필요시 여기서 SentencePiece, BPE 등으로 변경 가능.
    """
    return text.split()


def convert_txt_to_json(txt_file, json_file):
    data = []
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Converting to JSON"):
            line = line.strip()
            if line:  # 빈 줄은 무시
                tokens = tokenize_text(line)
                data.append({"tokens": tokens})

    with open(json_file, "w", encoding="utf-8") as json_f:
        for item in data:
            json_f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    txt_file = "chat_data_temp.txt"  # 원본 텍스트 파일 경로
    json_file = "chat_data.json"  # 변환 후 저장할 JSON 파일 경로
    convert_txt_to_json(txt_file, json_file)
    print(f"Conversion completed. JSON saved as {json_file}")
