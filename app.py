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

# Configuration de la page
st.set_page_config(
    page_title="Recherche et Analyse de Poèmes avec LLM", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Insérer directement la clé API ici (c'est temporaire et non recommandé pour la production)
api_key = "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"

# Vérifier si la clé est bien présente
if not api_key:
    st.error("Clé API OpenAI manquante. Assurez-vous de l'insérer correctement dans le fichier app.py.")
else:
    st.success("Clé API OpenAI chargée avec succès.")

# Initialiser le modèle OpenAI
llm_stream = ChatOpenAI(
    api_key=api_key,  # Utiliser la clé API ici
    model_name="gpt-4",
    temperature=0.7,
    streaming=True,
)

# Initialisation de la session
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

# Quand l'utilisateur clique sur "Rechercher"
if prompt := st.chat_input("Thème du poème (ex. amour)"):
    linecount = st.number_input("Longueur des vers (en nombre de lignes, optionnel)", min_value=1, step=1)
    st.session_state.messages.append({"role": "user", "content": f"Recherche de poèmes sur le thème '{prompt}' avec {linecount} vers."})

    # Recherche de poèmes via PoetryDB
    poems = get_poems_from_poetrydb(theme=prompt, linecount=linecount)

    if poems:
        for poem in poems:
            with st.chat_message("assistant"):
                st.markdown(f"**{poem['title']}**\n\n" + "\n".join(poem['lines']))
                st.session_state.messages.append({"role": "assistant", "content": f"**{poem['title']}**\n\n" + "\n".join(poem['lines'])})

        # Utilisation du LLM pour fournir une analyse littéraire
        with st.chat_message("assistant"):
            user_prompt = f"Voici un poème sur {prompt} avec {linecount} vers. Peux-tu en faire une analyse littéraire ?"
            messages = [HumanMessage(content=user_prompt)]

            full_response = ""
            placeholder = st.empty()  # Créer un espace réservé pour accumuler et afficher les résultats

            for chunk in llm_stream.stream(messages):
                full_response += chunk.content  # Ajout du texte sans espaces supplémentaires
                # Ajout d'une mise en forme du poème avec deux sauts de ligne pour séparer les strophes
                formatted_text = full_response.replace('\n', '  \n')  
                placeholder.markdown(formatted_text)  # Affichage du texte accumulé progressivement avec la bonne mise en forme

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
        # Vérification que le prompt n'est pas vide
        if prompt:
            messages = [HumanMessage(content=prompt)]
            full_response = ""
            placeholder = st.empty()  # Espace réservé pour la réponse

            # Itération sur le générateur pour afficher la réponse du LLM
            for chunk in stream_llm_response(llm_stream, messages):
                full_response += chunk  # Ajout du chunk sans espace supplémentaire
                # Ajout de la mise en forme avec des retours à la ligne appropriés
                formatted_text = full_response.replace('\n', '  \n')
                placeholder.markdown(formatted_text)  # Affichage du chunk dans l'interface

            # Enregistrement de la réponse complète dans l'état de session
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Veuillez saisir un message avant de soumettre.")
