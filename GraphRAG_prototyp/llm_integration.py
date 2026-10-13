from ollama import chat
from graph import verify
from graph import G
user_query = input("Bitte geben Sie Ihre Frage ein: ")
ll = verify(user_query, G)
response = chat(
    model="llama3.1",
    messages=[
        {f"role": "user", "content": "Du bist ein KI-Assistent, der die Wahrheit von Behauptungen überprüft. Bitte beantworte die folgende Frage basierend auf den Informationen:" + ll + "du sollst dich nur auf die dir gegbenen informationen stützen und nicht deine eigene meinung dazu geben. Bitte antworte in Deutsch. falls es uneinigkeiten in den informationen gibt sag das transparent und nenne die quellen und sage welche jeweils was behauptet und welche autorität die quellen haben. Formuliere die antwort ganz kurz. und am schluss der antwort sage die antwort anhand der quelle mit der höchsten autorität."},
    ]
)
print(response["message"]["content"])
#funktioniert nicht nur test