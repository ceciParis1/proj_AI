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

# Configuration de la page avec des éléments visuels
st.set_page_config(
    page_title="Recherche et Analyse de Poèmes avec LLM", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Titre principal avec style
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Recherche et Analyse de Poèmes avec LLM</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Ajout de texte d'introduction
st.markdown("""
<div style='text-align: center;'>
    <p>Utilisez cette application pour trouver des poèmes selon un thème et les analyser grâce à un modèle de langage.</p>
    <p style='color: #6c757d;'>Propulsé par GPT-4 et PoetryDB</p>
</div>
""", unsafe_allow_html=True)

# Insérer directement la clé API ici (c'est temporaire et non recommandé pour la production)
api_key = "\documentclass{beamer}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{xcolor}

% Thème et couleurs
\usetheme{Madrid} % Thème Beamer Madrid
\usecolortheme{seagull} % Schéma de couleurs neutres

% Couleurs personnalisées
\definecolor{myblue}{RGB}{0,102,204}
\setbeamercolor{structure}{fg=myblue}
\setbeamercolor{frametitle}{fg=myblue}
\setbeamercolor{title}{fg=myblue}

% Supprimer le pied de page avec le titre
\setbeamertemplate{footline}{}

% Titre de la présentation
\title{Recherche et analyse de poèmes avec LLM}
\subtitle{Une solution IA pour l'analyse littéraire automatisée}
\author{RUSSO Cécilia & DUPUIS Marine}
\institute{Paris 1 - Panthéon Sorbonne}
\date{Octobre 2024}

\begin{document}

% Diapositive de titre
\begin{frame}
    \titlepage
\end{frame}

% Table des matières
\begin{frame}{Sommaire}
    \tableofcontents
\end{frame}

% Section Introduction
\section{Introduction}

\begin{frame}{Contexte et Objectifs}
    \textbf{Contexte :} L'intelligence artificielle permet désormais de transformer la manière d'analyser les textes littéraires.
    
    \vspace{0.5cm}
    
    \textbf{Objectif :} Utiliser \textbf{GPT-4} et \textbf{PoetryDB} pour offrir une application permettant la recherche et l'analyse automatisée de poèmes.
    
    \vspace{0.5cm}
    
    \textbf{Lien vers l'application :} \href{https://projai-dwqgjwsgzzvgk2dbx39dyn.streamlit.app/}{Application Streamlit}
\end{frame}

% Section Fonctionnalités Clés
\section{Fonctionnalités}

\begin{frame}{Fonctionnalités de l'Application}
    \begin{itemize}
        \item Recherche de poèmes via \textbf{PoetryDB} en fonction du thème ou du nombre de lignes.
        \item Analyse littéraire automatisée avec \textbf{GPT-4}.
        \item Interface utilisateur simple et fluide avec \textbf{Streamlit}.
    \end{itemize}
    
    \vspace{0.5cm}
    
    \textbf{Avantage :} Adaptation pour des utilisateurs académiques ou professionnels (éducation, édition).
\end{frame}

% Section Interface Utilisateur avec Images
\section{Interface Utilisateur}

\begin{frame}{Interface Utilisateur - Capture 1}
    Voici une capture d'écran de l'application montrant l'interface principale :
    \begin{figure}[ht]
        \centering
        \includegraphics[width=0.8\textwidth]{/mnt/data/Image1.png}
        \caption{Interface principale de l'application}
    \end{figure}
\end{frame}

\begin{frame}{Interface Utilisateur - Capture 2}
    Exemple d'une recherche de poèmes avec un thème spécifique :
    \begin{figure}[ht]
        \centering
        \includegraphics[width=0.8\textwidth]{/mnt/data/Image2.png}
        \caption{Recherche de poèmes sur le thème des fleurs}
    \end{figure}
\end{frame}

\begin{frame}{Interface Utilisateur - Analyse Littéraire}
    Fonctionnalités de génération et d'analyse de poèmes par l'IA :
    \begin{figure}[ht]
        \centering
        \includegraphics[width=0.8\textwidth]{/mnt/data/Image3.png}
        \caption{Liste des capacités de l'IA en analyse de poèmes}
    \end{figure}
\end{frame}

\begin{frame}{Génération de Poèmes}
    Voici un exemple de génération de poèmes par l'application :
    \begin{figure}[ht]
        \centering
        \includegraphics[width=0.8\textwidth]{/mnt/data/Image4.png}
        \caption{Poème généré inspiré par Alfred de Musset}
    \end{figure}
\end{frame}

\begin{frame}{Exemple de Poèmes Célèbres}
    La capacité de retrouver des poèmes célèbres comme celui de Victor Hugo :
    \begin{figure}[ht]
        \centering
        \includegraphics[width=0.8\textwidth]{/mnt/data/Image5.png}
        \caption{Poème célèbre de Victor Hugo : Demain, dès l'aube}
    \end{figure}
\end{frame}

\begin{frame}{Biographies de Poètes}
    Exemple de la fonctionnalité pour obtenir des biographies :
    \begin{figure}[ht]
        \centering
        \includegraphics[width=0.8\textwidth]{/mnt/data/Image6.png}
        \caption{Biographie de Charles Baudelaire}
    \end{figure}
\end{frame}

% Section Méthodologie Technique
\section{Méthodologie Technique}

\begin{frame}{Architecture et Technologies Utilisées}
    \textbf{Technologies :}
    \begin{itemize}
        \item \textbf{GPT-4} : Génération d'analyses textuelles.
        \item \textbf{PoetryDB} : Base de données pour la recherche de poèmes.
        \item \textbf{Streamlit} : Interface web simple pour l'utilisateur.
        \item \textbf{Langchain} : Gestion des appels au modèle GPT-4.
    \end{itemize}
    
    \vspace{0.5cm}
    
    \textbf{Flux de Travail :} Saisie de thème → Recherche de poèmes → Analyse via GPT-4 → Affichage des résultats.
\end{frame}

% Section Avantages Commerciaux
\section{Opportunités Commerciales}

\begin{frame}{Potentiel Commercial}
    \begin{itemize}
        \item \textbf{Secteur éducatif :} Outil d'analyse pour les écoles et universités.
        \item \textbf{Éditeurs :} Analyses automatisées pour les maisons d'édition.
        \item \textbf{Expérience utilisateur :} Interaction intuitive pour les amateurs de poésie et les chercheurs littéraires.
    \end{itemize}
    
    \vspace{0.5cm}
    
    \textbf{Lien vers l'application :} \href{https://projai-dwqgjwsgzzvgk2dbx39dyn.streamlit.app/}{Application Streamlit}
\end{frame}

% Section Conclusion et Perspectives
\section{Conclusion et Perspectives}

\begin{frame}{Conclusion et Prochaines Étapes}
    \textbf{Conclusion :}
    \begin{itemize}
        \item Application innovante combinant recherche poétique et analyse IA.
        \item Déjà applicable dans plusieurs secteurs (éducation, édition).
    \end{itemize}

    \vspace{0.5cm}
    
    \textbf{Perspectives :}
    \begin{itemize}
        \item Intégration de nouveaux types de textes (narratifs, essais).
        \item Expansion linguistique pour couvrir plusieurs langues.
    \end{itemize}
\end{frame}

\end{document}
"

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

# Affichage des messages de la session avec mise en forme
for message in st.session_state.messages:
    role = "Utilisateur" if message["role"] == "user" else "Assistant"
    color = "#3498db" if message["role"] == "user" else "#2ecc71"
    st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:10px;color:white;'><strong>{role}:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Saisie d'un nouveau message par l'utilisateur
if prompt := st.chat_input("Votre message"):
    # Vérification si le message contient un thème pour la recherche de poèmes
    if "thème" in prompt.lower():
        # Extraire le thème du message
        theme = prompt.split("thème")[-1].strip()
        linecount = st.number_input("Longueur des vers (en nombre de lignes, optionnel)", min_value=1, step=1)
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
    else:
        # Enregistrement du message utilisateur
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
