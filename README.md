# Escale Atlantique — Sentiment des avis clients

Classe un avis client en **négatif, neutre ou positif** pour que le service qualité traite les clients mécontents en priorité. Le modèle [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment) prédit une note de 1 à 5 étoiles ; l'API la convertit en 3 classes.

## Architecture

```
                 docker compose (réseau interne "fastia")
 ┌──────────────────────────┐   POST /predict   ┌──────────────────────────┐
 │  ui  (Streamlit) :8501   │ ────────────────▶ │  api  (FastAPI) :8008    │
 │  un avis / lot CSV       │ ◀──────────────── │  modèle camembert (CPU)  │
 └──────────────────────────┘   sentiment +     └─────────┬────────────────┘
          ▲                     scores étoiles           │
          │ navigateur                        ┌──────────┴──────────┐
     service qualité                          │ volumes             │
                                              │ ./models  cache HF  │
                                              │ ./logs    api.log   │
                                              │ ./data    avis CSV  │
                                              └─────────────────────┘
```

- `api/` : FastAPI, charge le modèle au démarrage, expose `/health`, `/info`, `/predict`.
- `ui/` : Streamlit, un onglet pour un avis, un onglet pour un fichier CSV d'avis.
- `scripts/evaluer_avis.py` : fait passer les 30 avis annotés dans l'API et mesure.
- `data/avis_annotes.csv` : 30 avis annotés par l'équipe qualité.

## Installation, lancement, tests

```bash
cp .env.example .env
docker compose up --build        # 1er lancement : téléchargement du modèle (≈ 270 Mo)
```

| Quoi | Où |
|---|---|
| Interface | http://localhost:8501 |
| API : santé, doc | http://localhost:8008/health · http://localhost:8008/docs |
| Logs (une ligne par appel, texte tronqué à 80 caractères) | `logs/api.log` |

```bash
docker compose exec api pytest -v          # 22 tests : cas valide, 422, avis annotés, troncature des logs
python3 scripts/evaluer_avis.py            # mesure sur les 30 avis (API lancée)
docker compose down
```

## Conversion 5 étoiles → 3 classes

Le modèle renvoie une probabilité par étoile. On calcule la **note moyenne pondérée** (somme étoile × probabilité, de 1,0 à 5,0), puis : note < 2,5 → négatif, note > 3,5 → positif, neutre entre les deux.

Pourquoi pas l'étoile la plus probable ? Elle ignore la répartition : 40 % sur « 4 étoiles » et 60 % réparti sur 1 et 2 étoiles serait classé positif, alors que la note moyenne vaut 2,45. Ce cas est couvert par un test.

Résultat sur les 30 avis annotés (`scripts/evaluer_avis.py`) : **25 bien classés**, 2 négatifs ratés.

| attendu \ prédit | négatif | neutre | positif |
|---|---|---|---|
| négatif | 10 | 0 | 2 |
| neutre | 0 | 6 | 2 |
| positif | 1 | 0 | 9 |

| Conversion | Bien classés | Négatifs ratés |
|---|---|---|
| Note moyenne pondérée (retenue) | 25 | 2 |
| Étoile la plus probable | 25 | 2 |
| Prudente (négatif si note < 3,0) | 22 | 2 |

Les deux premières font les mêmes erreurs sur ce jeu ; la note moyenne reste retenue pour sa robustesse sur les distributions partagées. La variante prudente n'évite aucun négatif raté et crée 3 faux négatifs : les seuils ne sont pas le levier.

## Analyse des erreurs

Les 5 erreurs, avec la note moyenne calculée par le modèle :

| Id | Attendu → prédit | Note | Hypothèse |
|---|---|---|---|
| 3 | négatif → positif | 4,70 | **Ironie.** « Bravo pour l'insonorisation », « Merci pour l'expérience » : les mots sont positifs, le fait rapporté (voisins jusqu'à 3 h du matin) est négatif. |
| 18 | négatif → positif | 4,26 | **Ironie.** « Super, la vue mer donnait sur le parking. » Même mécanisme : le modèle lit le ton, pas la situation. |
| 17 | positif → négatif | 2,09 | **Négation.** « Pas une seule fausse note, on ne pouvait pas rêver mieux » : deux négations qui inversent des mots négatifs. |
| 22 | neutre → positif | 4,44 | **Vocabulaire et texte trop court.** « RAS. » est un jargon (rien à signaler) sans autre signal ; le modèle y voit une absence de plainte. |
| 30 | neutre → positif | 3,55 | **Avis mitigé, à la frontière.** Une phrase positive, une négative ; la note tombe juste au-dessus du seuil 3,5. |

Aucune erreur ne vient d'une **comparaison** : les avis 8 et 21 (« la moins bonne année », « depuis la rénovation ») sont bien classés.

**À retenir** : les deux erreurs graves (avis 3 et 18) sont des avis ironiques notés très positifs. Aucun réglage des seuils ne les rattrape sans casser les vrais positifs. Le levier est ailleurs : un modèle plus robuste à l'ironie, ou une relecture humaine des avis courts et de ceux dont la note est proche d'un seuil.
