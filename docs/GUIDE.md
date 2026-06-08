# Guide de l'équipe

Tout ce qu'il faut pour démarrer, développer et tester le projet. Pour la conception détaillée,
voir [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. Mise en route (5 min)

```bash
# Cloner puis, à la racine du dépôt :
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt                   # de quoi lancer les tests
python -m pytest -q                                    # doit afficher "23 passed"
```

> ⚠️ Utiliser `python -m pytest` (et non `pytest` seul) pour être sûr d'utiliser le bon
> interpréteur Python (celui de l'environnement virtuel).

À ce stade, **aucun modèle lourd n'est nécessaire** : les tests simulent FinBERT/Wav2Vec2/DeepFace.

## 2. Exécuter le système complet

Deux options, contrats REST identiques :

**A. Sans Docker (le plus simple, idéal Colab GPU)**
```bash
# installer les dépendances de chaque service + données (voir README)
python data/prepare_data.py     # télécharge les données (réseau + ffmpeg requis)
python run_all.py               # lance les 5 services
```

**B. Docker Compose**
```bash
python data/prepare_data.py
docker compose up --build
```

Tester la gateway :
```bash
curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
     -d '{"event_id":"2023-03-22"}'
```

## 3. Le rapport

Ouvrir `report.ipynb` (depuis la racine du dépôt) et exécuter les cellules : il interroge la
gateway pour chaque événement, affiche la table de corrélation et sauvegarde les nuages de points
dans `report_correlations.png`. La logique d'analyse est dans `analysis.py` (réutilisée par les tests).

## 4. Qui fait quoi (répartition possible)

| Domaine | Fichiers | Indépendant ? |
|---------|----------|----------------|
| Canal texte | `nlp-service/main.py` | oui |
| Canal audio | `audio-service/main.py` | oui |
| Canal vidéo | `vision-service/main.py` | oui |
| Fusion | `fusion-service/` | oui |
| Gateway / orchestration | `gateway-service/` | dépend des contrats |
| Données & marché | `data/` | oui |
| Analyse / rapport | `analysis.py`, `report.ipynb` | oui |
| Déploiement | `Dockerfile`, `docker-compose.yml`, `run_all.py` | oui |

Le **contrat partagé** est `shared/schema.py` (`ChannelScore`, `FusionResult`). Tant qu'on le
respecte, chaque canal se développe et se teste isolément.

## 5. Workflow de contribution

1. Créer une branche : `git checkout -b feat/mon-sujet`
2. Écrire le test d'abord (TDD), puis le code, jusqu'à ce que `python -m pytest -q` passe.
3. Commits clairs en *conventional commits* : `feat: …`, `fix: …`, `docs: …`, `test: …`.
4. Ouvrir une *pull request* vers `main`.

## 6. Conventions

- **Score** : valence `∈ [-1, 1]` sur le même axe pour tous les canaux (−1 négatif … +1 positif).
- **Imports paresseux** : ne jamais importer `transformers`/`torch`/`deepface`/`cv2` au niveau
  module d'un service — toujours à l'intérieur des fonctions, sinon les tests deviennent lents et
  exigent les modèles.
- **Tests sans réseau ni modèle** : on simule (`monkeypatch`) la fonction de score et le client HTTP.
- **GPU** : ne pas coder `cuda` en dur, utiliser `shared/device.py` (`get_device()`).

## 7. Dépannage

| Symptôme | Cause probable | Solution |
|----------|----------------|----------|
| `ModuleNotFoundError: shared` | lancé hors de la racine | lancer depuis la racine du dépôt |
| `pytest` ne trouve pas pydantic | mauvais Python | utiliser `python -m pytest`, activer le venv |
| `market_signal.json` vaut `null` | données marché manquantes/jour férié | vérifier la date, relancer `prepare_data.py` |
| vidéo `ok: false` | aucun visage détecté | normal (best-effort) ; la fusion continue sans la vidéo |
| audio muet | flux vidéo sans piste audio | `prepare_data.py` force déjà la fusion audio+vidéo ; revérifier l'URL |
