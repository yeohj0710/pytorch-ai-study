## pytorch AI study & practice

### /chatbot

- kakaotalk_chatbot.py
  - 카카오톡 채팅방 창을 오른쪽 아래에 붙인 뒤 실행하면 새로 올라오는 채팅을 파악하여 RNN 모델이 답변함
  - 새 채팅에는 반드시 "깡통"이라는 키워드가 포함되어 있어야 답변함
  - chatbot.pth라는 학습된 모델이 필요함
  - 학습 모델은 chat_data.txt를 이용해 test_chatbot.py로 학습시키면 생성됨
- 학습 data
  - 각 줄에 하나의 메시지 content가 저장된 chat_data.txt를 준비
