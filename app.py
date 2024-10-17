import streamlit as st
import os
import dotenv
import uuid
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from rag_methods import (
    get_poems_from_poetrydb,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

# Récupérer la clé API OpenAI
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Clé API OpenAI manquante. Assurez-vous qu'elle est définie dans les secrets Streamlit Cloud.")

# Initialiser le modèle OpenAI
llm_stream = ChatOpenAI(
    api_key=api_key,  # Utilisation de la clé API
    model_name="gpt-4",  # Modèle GPT-4
    temperature=0.7,
    streaming=True,
)

# Configuration de la page
st.set_page_config(
    page_title="Recherche et Analyse de Poèmes avec LLM", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Initialisation
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bienvenue ! Je peux vous aider à trouver des poèmes et à en discuter ou les analyser."}
    ]

# Affichage des messages de la session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Sidebar pour la recherche de poèmes
with st.sidebar:
    st.header("Rechercher des poèmes")
    theme = st.text_input("Thème du poème (optionnel)")
    linecount = st.number_input("Longueur des vers (en nombre de lignes, optionnel)", min_value=1, step=1)
    search_button = st.button("Rechercher des poèmes")

# Quand l'utilisateur clique sur "Rechercher"
if search_button:
    st.session_state.messages.append({"role": "user", "content": f"Recherche de poèmes sur le thème '{theme}' avec {linecount} vers."})

    # Recherche de poèmes via PoetryDB
    poems = get_poems_from_poetrydb(theme=theme, linecount=linecount)

    if poems:
        for poem in poems:
            with st.chat_message("assistant"):
                st.markdown(f"**{poem['title']}**\n\n" + "\n".join(poem['lines']))
                st.session_state.messages.append({"role": "assistant", "content": f"**{poem['title']}**\n\n" + "\n".join(poem['lines'])})

        # Utilisation du LLM pour fournir une analyse littéraire
        with st.chat_message("assistant"):
            user_prompt = f"Voici un poème sur {theme} avec {linecount} vers. Peux-tu en faire une analyse littéraire ?"
            messages = [HumanMessage(content=user_prompt)]
            
            full_response = ""
            for chunk in llm_stream.stream(messages):
                full_response += chunk.content
                st.markdown(chunk.content)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Aucun poème trouvé avec les critères spécifiés.")
        st.session_state.messages.append({"role": "assistant", "content": "Aucun poème trouvé avec les critères spécifiés."})

# Saisie d'un nouveau message par l'utilisateur
if prompt := st.chat_input("Votre message"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse du LLM
    with st.chat_message("assistant"):
        messages = [HumanMessage(content=prompt)]
        response = stream_llm_response(llm_stream, messages)
        st.markdown(response)
