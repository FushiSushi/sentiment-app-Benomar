
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permet d'importer le module de prédiction existant
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict


st.set_page_config(
    page_title="Analyse de sentiments",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Analyse de sentiments")
st.write("Analysez le sentiment d'un avis client grâce à notre modèle IA.")

tab1, tab2, tab3 = st.tabs(
    ["Un avis", "Un fichier CSV", "Comment ça marche"]
)

with tab1:
    st.subheader("Analyser un avis client")
    texte = st.text_area(
        "Saisissez votre avis :",
        placeholder="Ex. : Livraison rapide, très satisfait du produit...",
    )

    if st.button("Analyser le sentiment"):
        if not texte.strip():
            st.warning("Veuillez saisir un avis avant de lancer l'analyse.")
        else:
            try:
                resultat = predict(texte)
                label = resultat["label"]
                score = resultat["score"]

                if label == "positive":
                    st.success("Sentiment : Positif 😊")
                else:
                    st.error("Sentiment : Négatif 🙁")

                st.metric("Confiance du modèle", f"{score:.0%}")
                st.progress(score)

                if score < 0.60:
                    st.warning("À vérifier par un agent")

            except FileNotFoundError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.warning(str(exc))

with tab2:
    st.subheader("Analyser un fichier CSV")
    st.write(
        "Importez un CSV contenant une colonne appelée "
        "`texte` avec les avis clients."
    )

    fichier = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            df = pd.read_csv(fichier)

            if "texte" not in df.columns:
                st.error("Le fichier doit contenir une colonne 'texte'.")
            elif df.empty:
                st.warning("Le fichier ne contient aucun avis.")
            else:
                df = df.copy()
                df["texte"] = df["texte"].fillna("").astype(str)

                if st.button("Analyser le fichier"):
                    resultats = []
                    barre = st.progress(0)
                    statut = st.empty()

                    for i, texte_csv in enumerate(df["texte"]):
                        if texte_csv.strip():
                            try:
                                resultat = predict(texte_csv)
                            except (ValueError, FileNotFoundError):
                                resultat = {"label": "erreur", "score": 0.0}
                        else:
                            resultat = {"label": "erreur", "score": 0.0}

                        resultats.append(resultat)
                        barre.progress((i + 1) / len(df))
                        statut.text(f"Analyse : {i + 1}/{len(df)}")

                    df["sentiment"] = [r["label"] for r in resultats]
                    df["confiance"] = [r["score"] for r in resultats]

                    st.session_state["resultats_csv"] = df

        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Impossible de lire le fichier CSV : {exc}")

    if "resultats_csv" in st.session_state:
        resultat_df = st.session_state["resultats_csv"]

        st.subheader("Résultats")
        st.metric("Nombre d'avis", len(resultat_df))

        comptes = resultat_df["sentiment"].value_counts()
        st.bar_chart(comptes)

        st.subheader("Avis négatifs en priorité")
        st.dataframe(
            resultat_df.sort_values(
                by="sentiment",
                key=lambda col: col.map({"negative": 0, "positive": 1}).fillna(2),
            ),
            use_container_width=True,
        )

        st.download_button(
            "Télécharger les résultats CSV",
            data=resultat_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="resultats_sentiments.csv",
            mime="text/csv",
        )

with tab3:
    st.subheader("Comment ça marche ?")
    st.markdown(
        """
        1. **Saisir un avis** ou importer un fichier CSV.
        2. **Le modèle analyse le texte** et prédit son sentiment.
        3. **Consulter les résultats** et le niveau de confiance.
        4. Si la confiance est inférieure à 60 %, l'avis est signalé
           pour vérification par un agent.
        """
    )
