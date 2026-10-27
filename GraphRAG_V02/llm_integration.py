from ollama import chat
from graph import verify
from graph import G
user_query = input("Bitte geben Sie Ihre Frage ein: ")
ll = verify(user_query, G)
if "keine Evidenz im Graphen" in ll:
    print(ll)
else:
    response = chat(
        model="llama3.1",
        messages=[
            {f"role": "user", "content": "Du bist ein KI-Assistent, der die Wahrheit von Behauptungen überprüft. Bitte beantworte die folgende Frage basierend auf den Informationen:" + ll + "du sollst dich nur auf die dir gegbenen informationen stützen und nicht deine eigene meinung dazu geben. Bitte antworte in Deutsch. falls es uneinigkeiten in den informationen gibt sag das transparent und nenne die quellen und sage welche jeweils was behauptet und welche autorität die quellen haben. Formuliere die antwort ganz kurz. Das Verdikt steht schon fest, übernimm es so wie es ist und nutze die quellen als Begründung."},
        ]
    )
    print(response["message"]["content"])
