"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import os
from collections.abc import Callable

import httpx
import pandas as pd
import streamlit as st

# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8008")
TIMEOUT = 10
# Limite d'entrée de l'API (cf. api/app/schemas.py)
MAX_CARACTERES = 2000
# Garde-fou pour le traitement par lot : au-delà, l'attente devient trop longue pour une page web
MAX_AVIS_PAR_LOT = 500
# Colonnes attendues dans un export comme data/avis_annotes.csv
COLONNE_TEXTE = "texte"
COLONNE_ATTENDU = "sentiment_attendu"

# Les clients mécontents en premier : c'est la priorité du service qualité
SENTIMENTS = {
    "négatif": ("😠", "red"),
    "neutre": ("😐", "orange"),
    "positif": ("😊", "green"),
}
CLASSES = list(SENTIMENTS)


# --- Appel de l'API --------------------------------------------------------------

class ErreurApi(Exception):
    """Erreur affichable à l'utilisateur (avis refusé, erreur serveur…)."""


class ServiceIndisponible(ErreurApi):
    """Panne du service (réseau, délai, erreur 5xx) : inutile de continuer un lot."""


def appeler_api(texte: str, client: httpx.Client | None = None) -> dict:
    """Appelle /predict et renvoie le résultat, ou lève une ErreurApi.

    Les pannes réseau lèvent une exception httpx, les codes HTTP non : on gère les deux.
    Un `client` partagé évite d'ouvrir une connexion par avis lors d'un lot.
    """
    try:
        response = (client or httpx).post(f"{API_URL}/predict", json={"text": texte}, timeout=TIMEOUT)
    except httpx.ConnectError:
        raise ServiceIndisponible("Le service d'analyse est indisponible.")
    except httpx.TimeoutException:
        raise ServiceIndisponible("Le service met trop de temps à répondre.")
    except httpx.HTTPError as e:
        raise ServiceIndisponible(f"Erreur de communication avec l'API : {e}")

    if response.status_code == 200:
        return response.json()
    if response.status_code == 422:
        raise ErreurApi(f"L'avis doit contenir entre 1 et {MAX_CARACTERES} caractères.")
    if response.status_code >= 500:
        raise ServiceIndisponible("Erreur interne du serveur. Veuillez réessayer plus tard.")
    raise ErreurApi(f"Erreur lors de l'analyse du sentiment (code {response.status_code}).")


def note_moyenne(scores: dict[str, float]) -> float:
    """Note moyenne pondérée par les probabilités des 5 étoiles (de 1,0 à 5,0).

    Même formule que dans api/app/inference.py : l'API classe sur cette note.
    """
    return sum(int(label.split()[0]) * p for label, p in scores.items())


def analyser_lot(textes: list[str], sur_progres: Callable[[int, int], None]) -> list[dict]:
    """Analyse une liste d'avis un par un et renvoie une ligne de résultat par avis.

    Un avis refusé par l'API n'interrompt pas le lot : il est marqué dans sa colonne
    « erreur ». Une panne du service (ServiceIndisponible) remonte à l'appelant.
    """
    lignes = []
    with httpx.Client() as client:
        for i, texte in enumerate(textes, start=1):
            try:
                resultat = appeler_api(texte, client)
            except ServiceIndisponible:
                raise  # avant ErreurApi, dont elle hérite : une panne ne se note pas ligne par ligne
            except ErreurApi as e:
                lignes.append({"sentiment": "", "note_moyenne": None, "erreur": str(e)})
            else:
                lignes.append({
                    "sentiment": resultat["sentiment"],
                    "note_moyenne": round(note_moyenne(resultat["scores_etoiles"]), 2),
                    "erreur": "",
                })
            sur_progres(i, len(textes))
    return lignes


def oublier_le_lot() -> None:
    """Un autre fichier ou une autre colonne : les résultats affichés ne correspondent plus."""
    st.session_state.pop("lot", None)


# --- Page ------------------------------------------------------------------------

st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨")
st.title("🏨 Sentiment des avis clients")
st.caption("Escale Atlantique · outil du service qualité")

onglet_un, onglet_lot = st.tabs(["Un avis", "Lot d'avis (CSV)"])

# --- Onglet 1 : un seul avis ------------------------------------------------------

with onglet_un:
    text = st.text_area(
        "Avis client",
        height=160,
        max_chars=MAX_CARACTERES,
        placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
    )

    if st.button("Analyser", type="primary"):
        try:
            with st.spinner("Analyse par camembert…"):
                result = appeler_api(text)
        except ErreurApi as e:
            st.error(str(e))
        else:
            emoji, couleur = SENTIMENTS.get(result["sentiment"], ("❓", "gray"))
            st.badge(f"Sentiment : {result['sentiment']}", icon=emoji, color=couleur)
            # L'API renvoie les scores des 5 étoiles ; on affiche l'étoile la plus probable
            scores = result["scores_etoiles"]
            etoile = max(scores, key=scores.get)
            st.write(f"Note la plus probable : {etoile} ({scores[etoile]:.0%})")
            st.caption(f"Latence : {result['latence_ms']} ms")

# --- Onglet 2 : un fichier CSV d'avis ----------------------------------------------

with onglet_lot:
    st.markdown(
        "Déposez un fichier **CSV** contenant une colonne d'avis (par exemple l'export "
        "`avis_annotes.csv`). Chaque avis est envoyé à l'API, un par un."
    )
    fichier = st.file_uploader("Fichier CSV", type="csv", on_change=oublier_le_lot)

    if fichier is not None:
        try:
            df = pd.read_csv(fichier)
        except Exception:
            st.error("Impossible de lire ce fichier : vérifiez qu'il s'agit bien d'un CSV.")
            st.stop()

        if df.empty:
            st.warning("Le fichier ne contient aucune ligne.")
            st.stop()

        colonnes = list(df.columns)
        defaut = colonnes.index(COLONNE_TEXTE) if COLONNE_TEXTE in colonnes else 0
        colonne_texte = st.selectbox(
            "Colonne contenant l'avis", colonnes, index=defaut, on_change=oublier_le_lot
        )
        st.caption(f"{len(df)} avis dans le fichier")

        if len(df) > MAX_AVIS_PAR_LOT:
            st.warning(f"Le lot est limité à {MAX_AVIS_PAR_LOT} avis : seuls les premiers seront analysés.")
            df = df.head(MAX_AVIS_PAR_LOT)

        if st.button("Analyser le lot", type="primary"):
            textes = df[colonne_texte].fillna("").astype(str).tolist()
            progression = st.progress(0.0, text="Analyse en cours…")
            try:
                lignes = analyser_lot(
                    textes,
                    lambda i, n: progression.progress(i / n, text=f"{i}/{n} avis analysés"),
                )
            except ServiceIndisponible as e:
                progression.empty()
                st.error(f"{e} Lot interrompu.")
                st.stop()
            progression.empty()
            # Conservé dans la session : le tableau survit au clic sur « Télécharger ».
            # Trié une fois pour toutes : l'écran et le CSV téléchargé ont le même ordre.
            st.session_state["lot"] = (
                pd.concat([df, pd.DataFrame(lignes)], axis=1)
                .sort_values("note_moyenne", na_position="last")
            )

        if "lot" in st.session_state:
            resultats = st.session_state["lot"]
            analyses = resultats[resultats["erreur"] == ""]
            en_erreur = len(resultats) - len(analyses)

            st.subheader("Résultats")
            comptes = analyses["sentiment"].value_counts().reindex(CLASSES, fill_value=0)
            for col, (classe, (emoji, _)) in zip(st.columns(3), SENTIMENTS.items()):
                col.metric(f"{emoji} {classe.capitalize()}s", int(comptes[classe]))
            if en_erreur:
                st.warning(f"{en_erreur} avis n'ont pas pu être analysés (voir la colonne « erreur »).")

            st.bar_chart(comptes)

            # Si le fichier est annoté, on mesure l'accord avec l'équipe qualité
            if COLONNE_ATTENDU in analyses.columns:
                accord = (analyses["sentiment"] == analyses[COLONNE_ATTENDU]).sum()
                st.info(f"Accord avec l'annotation : {accord}/{len(analyses)} avis")

            st.dataframe(resultats, use_container_width=True, hide_index=True)
            st.download_button(
                "Télécharger les résultats (CSV)",
                data=resultats.to_csv(index=False).encode("utf-8"),
                file_name="avis_analyses.csv",
                mime="text/csv",
            )
