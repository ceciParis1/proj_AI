import requests
import streamlit as st

def get_poems_from_poetrydb(theme=None, linecount=None):
    base_url = "https://poetrydb.org/"
    url = ""

    # Construire l'URL en fonction des paramètres fournis
    if theme and linecount:
        url = base_url + f"lines/{linecount};theme/{theme}"
    elif theme:
        url = base_url + f"theme/{theme}"
    elif linecount:
        url = base_url + f"linecount/{linecount}"
    else:
        url = base_url + "random"

    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de la requête: {response.status_code}")
        return []

# Fonction pour le streaming de la réponse LLM
def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content  # Renvoyer les chunks progressivement

    # Ajouter la réponse à l'historique des messages
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_llm_rag_response(llm_stream, messages):
    # Placeholder pour la gestion RAG (si nécessaire plus tard)
    pass
