import json

import requests

# url = "http://localhost:8000/chat/invoke"
# data = {"question": "你好，介绍一下你自己"}
# response = requests.post(url, json=data)
# print(response.json())

url = "http://localhost:8000/chat/stream"
data = {"question": "好吧"}
response = requests.post(url, json=data, stream=True)

for line in response.iter_lines(decode_unicode=True):
    if not line:
        continue
    if not line.startswith("data: "):
        continue
    payload = line.removeprefix("data: ").strip()
    if payload == "[DONE]":
        break
    print(json.loads(payload)["content"], end="", flush=True)
print()
