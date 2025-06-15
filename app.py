import os
import tempfile
import sqlite3

import streamlit as st
import pandas as pd

from rag.langchain import answer_question as answer_with_langchain
from rag.langchain import store_pdf_file as store_pdf_file_langchain
from rag.langchain import delete_file_from_store as delete_file_langchain

from rag.llamaindex import answer_question as answer_with_llamaindex
from rag.llamaindex import store_pdf_file as store_pdf_file_llamaindex
from rag.llamaindex import delete_file_from_store as delete_file_llamaindex


# === CONFIGURATION ===
st.set_page_config(page_title="Analyse de documents", page_icon="👋")

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

# === INITIALISATION BDD ===
def init_db():
    conn = sqlite3.connect("feedbacks.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rating TEXT,
            question TEXT,
            response TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# === SAUVEGARDE DU FEEDBACK ===
def save_feedback_to_db(rating, question, response):
    conn = sqlite3.connect("feedbacks.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedbacks (rating, question, response)
        VALUES (?, ?, ?)
    ''', (rating, question, response))
    conn.commit()
    conn.close()


def main():
    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA. Chargez vos fichiers, posez vos questions.")

    # Sélecteur de langue
    language = st.selectbox("Choisissez la langue de réponse", ["français", "anglais", "espagnol", "allemand"])

    # Choix du framework
    framework = st.radio("Framework d'indexation", ["langchain", "llamaindex"])

    # Choix du nombre de documents récupérés
    k = st.slider("Nombre de documents similaires à récupérer", min_value=1, max_value=10, value=3)

    # Téléversement de fichiers
    uploaded_files = st.file_uploader("Déposez vos fichiers ici", type=['pdf'], accept_multiple_files=True)

    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_kb:.2f}"
            })

            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())

                if framework == "langchain":
                    store_pdf_file_langchain(path, f.name)
                else:
                    store_pdf_file_llamaindex(path, f.name)

                st.session_state['stored_files'].append(f.name)

        df = pd.DataFrame(file_info)
        st.table(df)

    # Suppression de fichiers retirés
    current_files = {f['Nom du fichier'] for f in file_info}
    for fname in set(st.session_state['stored_files']) - current_files:
        st.session_state['stored_files'].remove(fname)
        if framework == "langchain":
            delete_file_langchain(fname)
        else:
            delete_file_llamaindex(fname)

    # Champ de question
    question = st.text_input("Votre question ici")

    if st.button("Analyser"):
        if framework == "langchain":
            response = answer_with_langchain(question, language=language, k=k)
        else:
            response = answer_with_llamaindex(question, language=language, k=k)

        st.text_area("Réponse du modèle", value=response, height=200)

        # Feedback section - updated
        st.markdown("### La réponse vous a-t-elle été utile ?")
        rating = st.radio("", ["👍 Oui", "👎 Non"], horizontal=True)
        comments = st.text_area("Commentaires (optionnel)", height=80)

        if rating:
            save_feedback_to_db(rating, question, response)
            st.success("Merci pour votre retour !")
    else:
        st.text_area("Réponse du modèle", value="", height=200)


if __name__ == "__main__":
    main()
