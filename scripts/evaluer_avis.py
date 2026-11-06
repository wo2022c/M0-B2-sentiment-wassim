"""Tâche 6 — Fait passer les 30 avis annotés dans l'API et compare deux conversions.

Usage (API lancée via docker compose) :
    python3 scripts/evaluer_avis.py                 # API sur http://localhost:8008
    API_URL=http://localhost:8002 python3 scripts/evaluer_avis.py

Ne dépend que de la bibliothèque standard : l'API fait l'inférence, le script ne fait
que convertir les scores des 5 étoiles de deux façons différentes et compter.
"""

import csv
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8008")
CSV_AVIS = Path(__file__).resolve().parents[1] / "data" / "avis_annotes.csv"
CSV_SORTIE = Path(__file__).resolve().parents[1] / "data" / "evaluation_30_avis.csv"

CLASSES = ["négatif", "neutre", "positif"]


# --- Les conversions comparées ---------------------------------------------------

def note_moyenne(scores: dict[str, float]) -> float:
    """Espérance de la note : somme(étoile × probabilité)."""
    return sum(int(label.split()[0]) * p for label, p in scores.items())


def conversion_moyenne(scores: dict[str, float]) -> str:
    """Conversion 1 (celle de l'API, cf. api/app/inference.py) : note moyenne pondérée,
    < 2,5 négatif, > 3,5 positif, neutre entre les deux."""
    note = note_moyenne(scores)
    if note < 2.5:
        return "négatif"
    if note > 3.5:
        return "positif"
    return "neutre"


def conversion_argmax(scores: dict[str, float]) -> str:
    """Conversion 2 : l'étoile la plus probable. 1-2 négatif, 3 neutre, 4-5 positif."""
    etoile = int(max(scores, key=scores.get).split()[0])
    if etoile <= 2:
        return "négatif"
    if etoile >= 4:
        return "positif"
    return "neutre"


def conversion_prudente(scores: dict[str, float]) -> str:
    """Conversion 3 : comme la 1, mais le seuil négatif remonte à 3,0 pour rater moins de
    clients mécontents, quitte à classer négatif des avis mitigés."""
    note = note_moyenne(scores)
    if note < 3.0:
        return "négatif"
    if note > 3.5:
        return "positif"
    return "neutre"


CONVERSIONS = {
    "1. note moyenne (API)": conversion_moyenne,
    "2. étoile la plus probable": conversion_argmax,
    "3. prudente (négatif < 3,0)": conversion_prudente,
}


# --- Appel de l'API --------------------------------------------------------------

def predire(texte: str) -> dict:
    requete = urllib.request.Request(
        f"{API_URL}/predict",
        data=json.dumps({"text": texte}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        return json.load(reponse)


# --- Mesures ---------------------------------------------------------------------

def matrice_confusion(attendus: list[str], predits: list[str]) -> dict[tuple[str, str], int]:
    return Counter(zip(attendus, predits))


def afficher_matrice(matrice: dict[tuple[str, str], int]) -> None:
    largeur = max(len(c) for c in CLASSES)
    print(f"  {'attendu \\ prédit':<{largeur + 10}}" + "".join(f"{c:>9}" for c in CLASSES))
    for attendu in CLASSES:
        ligne = "".join(f"{matrice.get((attendu, predit), 0):>9}" for predit in CLASSES)
        print(f"  {attendu:<{largeur + 10}}{ligne}")


def negatifs_rates(matrice: dict[tuple[str, str], int]) -> int:
    """Erreurs graves : avis négatifs classés neutre ou positif (client mécontent non traité)."""
    return sum(n for (attendu, predit), n in matrice.items() if attendu == "négatif" and predit != "négatif")


def main() -> int:
    with CSV_AVIS.open(encoding="utf-8", newline="") as f:
        avis = list(csv.DictReader(f))

    print(f"API : {API_URL}  —  {len(avis)} avis depuis {CSV_AVIS.name}\n")

    resultats = []
    for a in avis:
        reponse = predire(a["texte"])
        ligne = {
            "id": a["id"],
            "attendu": a["sentiment_attendu"],
            "api": reponse["sentiment"],
            "note_moyenne": round(note_moyenne(reponse["scores_etoiles"]), 2),
        }
        for nom, conversion in CONVERSIONS.items():
            ligne[nom] = conversion(reponse["scores_etoiles"])
        resultats.append(ligne)

    # Garde-fou : la conversion 1 recalculée ici doit être identique à celle de l'API
    ecarts = [r["id"] for r in resultats if r["1. note moyenne (API)"] != r["api"]]
    if ecarts:
        print(f"⚠ la conversion 1 diffère de l'API pour les avis {ecarts}", file=sys.stderr)

    attendus = [r["attendu"] for r in resultats]
    bilan = {}
    for nom in CONVERSIONS:
        predits = [r[nom] for r in resultats]
        matrice = matrice_confusion(attendus, predits)
        bien_classes = sum(a == p for a, p in zip(attendus, predits))
        bilan[nom] = (bien_classes, negatifs_rates(matrice))

        print(f"=== {nom} ===")
        print(f"  bien classés : {bien_classes}/{len(avis)}")
        print(f"  négatifs ratés (erreurs graves) : {negatifs_rates(matrice)}")
        afficher_matrice(matrice)
        erreurs = [f"{r['id']}({r['attendu']}→{r[nom]})" for r in resultats if r["attendu"] != r[nom]]
        print(f"  erreurs : {', '.join(erreurs) or 'aucune'}\n")

    print("=== Comparaison ===")
    for nom, (ok, graves) in bilan.items():
        print(f"  {nom:<30} bien classés : {ok:>2}   négatifs ratés : {graves}")

    with CSV_SORTIE.open("w", encoding="utf-8", newline="") as f:
        colonnes = ["id", "attendu", "note_moyenne", *CONVERSIONS]
        writer = csv.DictWriter(f, fieldnames=colonnes, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(resultats)
    print(f"\nDétail par avis écrit dans {CSV_SORTIE.relative_to(Path.cwd()) if CSV_SORTIE.is_relative_to(Path.cwd()) else CSV_SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
