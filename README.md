# Sentiment multimodal FOMC — Système réparti

La voix de Powell influence-t-elle les marchés ? Ce projet analyse les conférences de presse
du **FOMC** (Réserve fédérale américaine) selon trois canaux indépendants — **texte** (FinBERT),
**audio** (Wav2Vec2) et **vidéo** (DeepFace) — fusionne les trois signaux, puis compare chaque
canal et leur fusion à la variation du **S&P 500** après chaque conférence.

C'est un projet de **systèmes répartis** : 5 microservices FastAPI communiquant en REST,
orchestrés par une *gateway* et déployables via Docker Compose (ou en local/Colab).

📄 Documentation détaillée : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/GUIDE.md`](docs/GUIDE.md) (guide de l'équipe)

## 🚀 Démarrage rapide (local, sans Docker)

```bash
# 1. Se placer à la racine du projet
cd fomc-sentiment-system

# 2. Installer les dépendances (les 5 services + données)
pip install -r nlp-service/requirements.txt -r audio-service/requirements.txt \
            -r vision-service/requirements.txt -r gateway-service/requirements.txt \
            -r fusion-service/requirements.txt -r data/requirements.txt

# 3. Préparer les données (télécharge vidéos/audio/transcriptions + S&P 500)
python data/prepare_data.py

# 4. Démarrer les 5 services (laisser ce terminal ouvert ; Ctrl+C pour arrêter)
python run_all.py
```

5. Ouvrir l'interface dans un navigateur : **http://localhost:8000/**

Dans l'interface : *Texte libre* et *Corrélations* sont instantanés ; *Analyser un événement*
lance les 3 modèles en direct (**1 à 4 min sur CPU**, davantage au premier lancement le temps de
télécharger les modèles). Sans navigateur, on peut aussi appeler la gateway en ligne de commande :

```bash
curl -X POST localhost:8000/analyze -H 'content-type: application/json' -d '{"event_id":"2023-03-22"}'
python run_report.py     # régénère le dossier results/ (scores + corrélations + graphique)
```

> GPU détecté automatiquement (`shared/device.py`) : `cuda` si disponible, sinon `cpu`.
> Port déjà utilisé ? `pkill -9 -f run_all.py; pkill -9 -f "uvicorn main:app"`.

## Architecture en bref

```
                         ┌──────────────────┐
   client / notebook  →  │  gateway  (8000) │  orchestrateur
                         └───┬────┬────┬─────┘
              POST /analyze  │    │    │
          ┌──────────────────┘    │    └──────────────────┐
          ▼              ▼         ▼                        ▼
     nlp (8001)    audio (8002)  vision (8003)       fusion (8004)
     FinBERT       Wav2Vec2      DeepFace            combine les scores
     (texte)       (émotion voix) (émotion visage)   → signal de marché
```

La gateway interroge les trois canaux **en parallèle**, transmet leurs scores à `fusion`, et
renvoie le score combiné. Chaque service détecte automatiquement le GPU (`cuda`) ou le CPU.

## Périmètre honnête

L'échantillon est volontairement petit (3 à 5 événements, le plus simple à constituer). Les
corrélations sont **illustratives, non significatives statistiquement**. Pour ajouter des
événements : éditer `data/events.json` (aucun changement de code), puis relancer la préparation.

## Lancer en local (Docker)

```bash
pip install -r data/requirements.txt
python data/prepare_data.py            # télécharge vidéos/audio/transcriptions + données marché
docker compose up --build
curl -X POST localhost:8000/analyze -H 'content-type: application/json' -d '{"event_id":"2023-03-22"}'
```

## Lancer sur Colab (GPU, sans Docker)

```bash
pip install -r nlp-service/requirements.txt -r audio-service/requirements.txt \
            -r vision-service/requirements.txt -r gateway-service/requirements.txt \
            -r fusion-service/requirements.txt -r data/requirements.txt
python data/prepare_data.py
python run_all.py &
# puis exécuter report.ipynb
```

Le GPU est détecté automatiquement (`shared/device.py`) : `cuda` si disponible, sinon `cpu`.

## Interface de test

Une fois les services démarrés (`docker compose up` ou `python run_all.py`), ouvrir
**http://localhost:8000/** dans un navigateur. La page (servie par la gateway) permet de :

- **Analyser un événement** — choisir une conférence FOMC et lancer l'analyse multimodale en direct ;
  jauge de fusion + score de chaque canal (nlp / audio / vidéo) avec statut ok/échec.
- **Texte libre** — coller un texte et obtenir instantanément le score NLP (FinBERT).
- **Corrélations** — afficher la table des scores et le graphique sur tous les événements (`results/`).

Endpoints correspondants : `GET /` (UI), `GET /events`, `POST /analyze`, `POST /analyze_text`,
`GET /results`, `GET /results/correlations.png`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Les tests n'exigent aucun modèle lourd : les appels FinBERT/Wav2Vec2/DeepFace et HTTP sont
simulés. Les modèles réels ne tournent que lors de l'exécution end-to-end (cf. `docs/GUIDE.md`).

## Rapport

`report.ipynb` exécute le pipeline sur tous les événements et produit la table de corrélation
et les nuages de points (`report_correlations.png`). La logique d'analyse est dans `analysis.py`.
