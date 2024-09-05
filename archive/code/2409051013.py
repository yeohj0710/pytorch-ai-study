import re


def process_chat_log(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            # 빈 줄 제거
            line = line.strip()
            if line == "":
                continue

            # 날짜로 시작하는 줄에서 메시지만 추출
            match = re.match(
                r"^\d{4}년 \d{1,2}월 \d{1,2}일 (오전|오후) \d{1,2}:\d{2}, [^:]+: (.+)",
                line,
            )
            if match:
                message = match.group(2).strip()
                outfile.write(message + "\n")


if __name__ == "__main__":
    input_file = "chats.txt"
    output_file = "chat_data.txt"
    process_chat_log(input_file, output_file)
    print(f"Processed chat log saved to {output_file}")
