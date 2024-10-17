import streamlit as st
import os
import dotenv
import uuid
import requests
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

dotenv.load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Recherche et Analyse de Poèmes avec LLM", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Insérer directement la clé API ici (c'est temporaire et non recommandé pour la production)
api_key = "sk-zOGMVhGdlWR_RehPscj0d2KVx9Csi1S0gp_x8Rmt3GT3BlbkFJaqcM7GWiZmVnCSL4Mkm43wxIzQ2ADT1g1_WK1MoUUA"
openai.api_key = api_key  # Définir la clé API OpenAI pour l'utilisation d'embeddings

# Initialiser le modèle OpenAI
llm_stream = ChatOpenAI(
    api_key=api_key,
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

# Stockage des poèmes et leurs embeddings
poem_embeddings = []
poems_data = []

# Fonction pour récupérer des poèmes depuis PoetryDB
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

# Fonction pour obtenir l'embedding d'un texte
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

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
                
                # Obtenir l'embedding pour le poème
                poem_text = "\n".join(poem['lines'])
                embedding = get_embedding(poem_text)
                
                # Stocker l'embedding et les données du poème
                poem_embeddings.append(embedding)
                poems_data.append(poem)

        # Utilisation du LLM pour fournir une analyse littéraire
        with st.chat_message("assistant"):
            user_prompt = f"Voici un poème sur {prompt} avec {linecount} vers. Peux-tu en faire une analyse littéraire ?"
            messages = [HumanMessage(content=user_prompt)]

            full_response = ""
            placeholder = st.empty()  # Créer un espace réservé pour accumuler et afficher les résultats

            for chunk in llm_stream.stream(messages):
                full_response += chunk.content  # Ajout du texte sans espaces supplémentaires
                formatted_text = full_response.replace('\n', '  \n')  
                placeholder.markdown(formatted_text)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Aucun poème trouvé avec les critères spécifiés.")
        st.session_state.messages.append({"role": "assistant", "content": "Aucun poème trouvé avec les critères spécifiés."})

# Recherche de poèmes similaires
if prompt := st.chat_input("Chercher des poèmes similaires"):
    # Obtenir l'embedding pour la requête de l'utilisateur
    query_embedding = get_embedding(prompt)

    # Calculer les similarités
    similarities = cosine_similarity([query_embedding], poem_embeddings)

    # Trouver les indices des poèmes les plus similaires
    similar_poem_indices = np.argsort(similarities[0])[::-1][:5]  # Récupérer les 5 poèmes les plus similaires

    st.chat_message("assistant").markdown("Voici les poèmes similaires :")
    for index in similar_poem_indices:
        poem = poems_data[index]
        st.markdown(f"**{poem['title']}**\n\n" + "\n".join(poem['lines']))
        st.session_state.messages.append({"role": "assistant", "content": f"**{poem['title']}**\n\n" + "\n".join(poem['lines'])})

# Saisie d'un nouveau message par l'utilisateur
if prompt := st.chat_input("Votre message"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse du LLM
    with st.chat_message("assistant"):
        if prompt:
            messages = [HumanMessage(content=prompt)]
            full_response = ""
            placeholder = st.empty()  # Espace réservé pour la réponse

            for chunk in stream_llm_response(llm_stream, messages):
                full_response += chunk
                formatted_text = full_response.replace('\n', '  \n')
                placeholder.markdown(formatted_text)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Veuillez saisir un message avant de soumettre.")
