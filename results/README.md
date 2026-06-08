# Résultats

Résultats **réels** produits par le pipeline sur les 3 conférences de presse du FOMC
(transcriptions officielles de la Fed, vidéos de la chaîne Federal Reserve, S&P 500 via yfinance).

| Fichier | Contenu |
|---------|---------|
| `scores.csv` | Score par canal (nlp, audio, vision), score de fusion et variation du marché, par événement |
| `correlations.csv` | Corrélation de chaque canal avec la variation du S&P 500 |
| `correlations.png` | Nuages de points sentiment vs marché (un par canal) |

## Lecture des résultats (échantillon = 3 événements)

| Événement | nlp | audio | vision | fusion | marché (S&P %) |
|-----------|-----|-------|--------|--------|----------------|
| 2023-03-22 | −0.07 | 0.00 | −1.00 | −0.23 | +0.86 |
| 2023-05-03 | +0.09 | +0.31 | −0.04 | +0.13 | +1.11 |
| 2023-06-14 | +0.01 | +0.30 | +0.06 | +0.11 | +0.85 |

Corrélation avec le marché : **nlp 0.82**, audio 0.47, vision 0.37, fusion 0.49.

> ⚠️ **Non significatif.** Avec seulement 3 points, ces corrélations sont **illustratives**, pas une
> conclusion. Le canal vidéo de mars 2023 vaut −1.0 car DeepFace n'a détecté que des émotions
> négatives sur les images échantillonnées. Pour conclure, étendre `data/events.json` à 30+
> événements puis relancer.

## Régénérer les résultats

```bash
python data/prepare_data.py     # télécharge données (réseau + ffmpeg requis)
python run_all.py &             # démarre les 5 services (charge les modèles)
python run_report.py            # interroge la gateway et réécrit ce dossier
#   …ou exécuter report.ipynb
```
