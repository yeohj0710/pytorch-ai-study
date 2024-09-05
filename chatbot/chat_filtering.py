import re
import os


def is_korean(text):
    # 한글만 포함하는지 체크
    return all(
        "\uac00" <= char <= "\ud7a3"
        or char.isspace()
        or char in [".", ",", "?", "!", "ㅋㅋ", "ㅎ"]
        for char in text
    )


def filter_chat_lines(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    filtered_lines = []

    for line in lines:
        line = line.strip()

        # 1. 모든 문자가 자음이거나 단순한 문장 제거 (예: ㅋㅋㅋㅋ, ...., ? 등)
        if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+|[\.?!]+", line):
            continue

        # 2. 텍스트 길이가 20자가 넘어가는 문장 제거
        if len(line) > 20:
            continue

        # 3. '삭제된 메시지입니다.' 문장 제거
        if line == "삭제된 메시지입니다.":
            continue

        # 4. 'https://'로 시작하는 URL 제거
        if line.startswith("https://") or line.startswith("http://"):
            continue

        # 5. '샵검색'으로 시작하는 문장 제거
        if line.startswith("샵검색"):
            continue

        # 6. 한글이 아닌 영어, 일본어 등 다른 언어로 작성된 문장 제거
        if not is_korean(line):
            continue

        # 7. 사진이나 동영상, 이모티콘은 제외
        if line in ["사진", "동영상", "이모티콘"]:
            continue

        filtered_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as output_file:
        for line in filtered_lines:
            output_file.write(line + "\n")


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "chat_data.txt")

filter_chat_lines(file_path)
