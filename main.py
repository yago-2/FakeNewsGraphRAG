from ollama import chat


with open("artikel.txt") as f:
    text = f.read()
response = chat(
    model="llama3.1 ",
    messages=[
        {
            "role": "user",
            "content": f"Fasse diesen Text zusammen:{text}"
        }
    ]
)
print(response["message"]["content"])