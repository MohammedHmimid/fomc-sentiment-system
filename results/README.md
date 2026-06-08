# Résultats

Ce dossier contient **tous les résultats générés** par le pipeline.

| Fichier | Contenu |
|---------|---------|
| `scores.csv` | Score par canal (nlp, audio, vision), score de fusion et variation du marché, par événement |
| `correlations.csv` | Corrélation de chaque canal avec la variation du S&P 500 |
| `correlations.png` | Nuages de points sentiment vs marché (un par canal) |

## Comment (re)générer les résultats réels

```bash
python data/prepare_data.py     # télécharge les données (réseau + ffmpeg requis)
python run_all.py               # démarre les 5 services (charge les modèles)
# puis exécuter report.ipynb  → écrit scores.csv, correlations.csv, correlations.png ici
```

On peut aussi appeler directement `analysis.save_results(df)`.

## ⚠️ Fichiers `*_exemple.*`

Les fichiers suffixés `_exemple` sont générés à partir de **données factices** pour montrer le
format de sortie. **Ce ne sont PAS de vrais résultats.** Ils sont remplacés par les vrais fichiers
(`scores.csv`, `correlations.csv`, `correlations.png`) dès que le pipeline est exécuté.
