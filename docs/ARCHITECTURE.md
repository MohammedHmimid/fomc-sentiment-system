# Architecture

Document technique du système réparti de sentiment multimodal FOMC.

## Vue d'ensemble

Cinq microservices FastAPI indépendants, communiquant en **REST/HTTP**, reliés par un réseau
Docker Compose. Chaque service a une responsabilité unique et un contrat d'API clair.

```
                         ┌──────────────────┐
   client / notebook  →  │  gateway  (8000) │  orchestrateur
                         └───┬────┬────┬─────┘
              POST /analyze  │    │    │
          ┌──────────────────┘    │    └──────────────────┐
          ▼              ▼         ▼                        ▼
     nlp (8001)    audio (8002)  vision (8003)       fusion (8004)
     FinBERT       Wav2Vec2      DeepFace            fusion tardive des scores
```

## Les services

| Service | Port | Rôle | Modèle |
|---------|------|------|--------|
| `gateway`  | 8000 | Reçoit un `event_id`, appelle les 3 canaux **en parallèle**, transmet les scores à `fusion`, renvoie le résultat | — |
| `nlp`      | 8001 | Sentiment du **texte** de la transcription | FinBERT (`ProsusAI/finbert`) |
| `audio`    | 8002 | Émotion de la **voix** | Wav2Vec2 (`superb/wav2vec2-base-superb-er`) |
| `vision`   | 8003 | Émotion du **visage** (best-effort) | DeepFace |
| `fusion`   | 8004 | Combine les 3 scores (fusion tardive pondérée) | — |

Chaque service de canal isole son modèle derrière **une seule fonction** (`score_text`,
`score_audio`, `score_video`) à imports **paresseux** : importer `main.py` ne charge pas
transformers/torch/deepface. C'est ce qui permet aux tests de simuler le modèle sans
téléchargement, et de charger le GPU uniquement à la première requête.

## Contrat d'API

Tous les `/analyze` renvoient le même schéma (`shared/schema.py`), un `ChannelScore` :

```json
{ "event_id": "2023-03-22", "channel": "nlp",
  "score": -0.42, "label": "negative", "raw": { }, "ok": true, "error": null }
```

- `score ∈ [-1, 1]` est une **valence de sentiment** sur un axe unique pour tous les canaux
  (−1 = négatif, 0 = neutre, +1 = positif), ce qui les rend directement comparables.
- `ok: false` + `error` quand un canal ne peut pas produire de score.

`fusion` renvoie un `FusionResult` : `combined_score`, `label`, `channels` (les 3 `ChannelScore`),
`weights_used`, `channels_used`.

## Flux de données

```
data/prepare_data.py  ──►  data/processed/<event_id>/
   yt-dlp (vidéo)            ├── video.mp4      ← lu par vision
   ffmpeg (audio 16kHz)      ├── audio.wav      ← lu par audio
   transcription Fed         ├── transcript.txt ← lu par nlp
   yfinance (S&P 500)        ├── market.csv
                             └── market_signal.json  (variation % du marché)
```

À l'exécution : `gateway` lit l'`event_id`, les 3 services lisent leur fichier dans
`data/processed/<event_id>/`, renvoient un `ChannelScore`, et `fusion` agrège.

## Dégradation gracieuse

Le canal **vidéo** est le plus fragile (DeepFace peut ne trouver aucun visage). Il est
**best-effort** : en cas d'échec il renvoie `ok: false`, et `fusion` **renormalise les poids**
sur les canaux disponibles. De même, la gateway transforme une panne réseau d'un canal (ou de
`fusion`) en résultat structuré plutôt qu'en erreur 500. Une vidéo défaillante ne casse jamais
le pipeline.

## Fusion

Fusion **tardive au niveau des scores**, pas un modèle entraîné (l'échantillon est trop petit
pour entraîner quoi que ce soit). Poids par défaut : `nlp 0.5`, `audio 0.3`, `vision 0.2`,
renormalisés sur les canaux réussis. C'est ce qui permet de comparer proprement *texte seul* vs
*audio* vs *vidéo* vs *fusion*.

## Déploiement

- **Docker Compose** : un seul `Dockerfile` racine paramétré par les arguments `SERVICE` et
  `PORT` ; `docker-compose.yml` instancie les 5 services + le réseau + les volumes de données.
- **Colab / local sans Docker** : `run_all.py` lance les 5 services comme sous-processus uvicorn
  sur les mêmes ports — contrats REST identiques, le notebook fonctionne dans les deux cas.

## Arborescence

```
gateway-service/   main.py, orchestrator.py, requirements.txt
nlp-service/       main.py, requirements.txt        (FinBERT)
audio-service/     main.py, requirements.txt        (Wav2Vec2)
vision-service/    main.py, requirements.txt        (DeepFace)
fusion-service/    main.py, fusion.py, requirements.txt
shared/            schema.py (contrat), device.py (cuda/cpu)
data/              prepare_data.py, events.json, market.py, processed/<id>/
analysis.py        corrélations + graphiques (partagé notebook/tests)
report.ipynb       rapport comparatif
Dockerfile         image unique des 5 services
docker-compose.yml · run_all.py   déploiement
tests/             tests unitaires (modèles et HTTP simulés)
```

## Comment étendre

- **Ajouter des événements FOMC** : ajouter une entrée dans `data/events.json`
  (`id`, `video_url`, `transcript_url`, `fomc_datetime`), relancer `python data/prepare_data.py`.
  Aucun code à modifier.
- **Ajouter un 4ᵉ canal** : créer `<nom>-service/main.py` sur le même patron (fonction de score
  à import paresseux + endpoints `/health` et `/analyze` renvoyant un `ChannelScore`), l'ajouter
  à `gateway-service/orchestrator.py`, à `fusion` (poids), à `docker-compose.yml` et `run_all.py`.
