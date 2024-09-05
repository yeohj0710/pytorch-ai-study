import os

try:
    input_percentage = int(input("추출할 비율을 입력하세요 (0~100): "))
    if not (0 <= input_percentage <= 100):
        raise ValueError("입력 비율은 0에서 100 사이여야 합니다.")
except ValueError as e:
    print(f"잘못된 입력: {e}")
    exit()

script_dir = os.path.dirname(os.path.abspath(__file__))

input_file_path = os.path.join(script_dir, "chat_data.txt")
output_file_path = os.path.join(script_dir, "chat_data.txt")

try:
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"{input_file_path} 파일이 존재하지 않습니다.")
    exit()

total_lines = len(lines)

num_lines_to_extract = int(total_lines * input_percentage / 100)

lines_to_save = lines[-num_lines_to_extract:]

with open(output_file_path, "w", encoding="utf-8") as file:
    file.writelines(lines_to_save)

print(
    f"파일에서 {input_percentage}% ({num_lines_to_extract}줄) 추출하여 '{output_file_path}'로 저장하였습니다."
)
