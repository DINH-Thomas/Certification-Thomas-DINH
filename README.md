# Certification-Thomas-DINH

## Guide d’installation

### Pré-requis système minimaux
- Git
- Python 3.11
- Pip
- Optionnel mais recommandé: pyenv
- Docker et Docker Compose si vous voulez lancer l'application via conteneurs

### Installation locale
1. Créer un environnement virtuel Python 3.11.
2. Installer les dépendances avec `pip install -r requirements.txt`.
3. Copier `.env-example` en `.env` puis renseigner les variables nécessaires.

### Variables d’environnement
Le projet lit un fichier `.env` à la racine du dépôt. Le fichier `.env-example` liste les variables attendues pour un lancement local de base.

Variables importantes pour démarrer localement:
- `DATABASE_URL` pour la base SQLite locale.
- `API_URL_LOCAL` pour le dashboard.
- `GDRIVE_MODEL_FOLDER_ID` si vous voulez que l’API télécharge automatiquement les modèles manquants depuis Google Drive.
- `HEADERS` pour les scripts de collecte de données.

### Modèles
Le dépôt ne contient pas tous les artefacts de modèle. Pour un clone vierge, il faut soit:
- copier le dossier `models` complet à la racine du projet, avec les fichiers attendus par l’API.
- soit renseigner `GDRIVE_MODEL_FOLDER_ID` dans `.env` pour permettre le téléchargement automatique des modèles manquants au démarrage.

Les modèles sont disponibles ici: https://drive.google.com/drive/folders/14HIhHMygB9t0E66EMmORX2Bcd8SWAjME?hl=fr

### Lancement local
- Lancer l’API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- Lancer le dashboard: `streamlit run src/dashboard/app.py`

### Structure des fichiers Python
Voici un aperçu du rôle de chaque fichier `.py` du projet:

#### `src/api/`
- `main.py`: point d’entrée FastAPI, expose les routes `/predict`, `/explain`, `/stats` et initialise la base au démarrage.
- `database.py`: gère la connexion SQLAlchemy, l’initialisation de la base et l’historique des prédictions.
- `schemas.py`: définit les modèles Pydantic utilisés pour les requêtes et réponses de l’API.
- `services.py`: charge les modèles, calcule les prédictions et produit les explications par mot.

#### `src/config/`
- `config.py`: centralise les chemins, variables d’environnement et paramètres globaux du projet.
- `gdrive_loader.py`: télécharge automatiquement les artefacts de modèle manquants depuis Google Drive.

#### `src/data_cleaning/`
- `download_data.py`: télécharge le dataset Kaggle et scrape des subreddits positifs via Reddit.
- `clean_data.py`: nettoie les données brutes et fusionne les colonnes de texte utiles.
- `balance_classes.py`: combine les dataframes téléchargés et scrappés puis équilibre les classes pour obtenir un dataset final exploitable.

#### `src/training/`
- `preprocess.py`: normalise et tokenize les textes avant l’entraînement ou l’inférence.
- `train.py`: prépare les splits, entraîne les modèles et sauvegarde les artefacts.
- `evaluate.py`: calcule les métriques de performance sur le jeu de test.
- `predict.py`: exécute l’inférence pour les différents modèles supportés.

#### `src/pipeline/`
- `run_full_pipeline.py`: orchestre toute la chaîne, du téléchargement des données jusqu’à l’entraînement et l’évaluation.

#### `src/dashboard/`
- `app.py`: point d’entrée Streamlit du dashboard.
- `about.py`: présente les modèles et le contexte du projet.
- `examples.py`: fournit des exemples de textes à injecter dans les formulaires.
- `pages.py`: contient les pages principales de prédiction et d’explication.
- `shap.py`: calcule et affiche les éléments SHAP pour l’interprétabilité.
- `stats.py`: affiche les statistiques d’usage récupérées depuis l’API.

#### `tests/`
- `conftest.py`: configure les fixtures partagées par les tests.
- `data_cleaning/*.py`: vérifie le téléchargement, le nettoyage et l’équilibrage des données.
- `training/*.py`: vérifie le prétraitement, l’entraînement, l’évaluation et la prédiction.
- `pipeline/test_run_full_pipeline.py`: vérifie l’orchestration de la pipeline complète.
- `api/` et `dashboard/`: contiennent les tests dédiés aux couches API et interface.

### Pipeline complète (Kaggle + scraping + cleaning + training + métriques)
Pour exécuter toute la chaîne automatiquement en une seule commande:

`python -m src.pipeline.run_full_pipeline`

Cette commande:
- télécharge le dataset Kaggle `reddit-depression-dataset` (si absent),
- scrape les subreddits configurés dans `src/config/config.py`,
- nettoie les données Kaggle et scrapées,
- balance les classes et sauvegarde le dataset final dans `data/cleaned/balanced_30k_dataset.csv`,
- entraîne les modèles Logistic Regression et XGBoost,
- affiche les métriques `accuracy`, `precision`, `recall`, `f1` pour les deux modèles.

Option utile pour réduire le volume de scraping (tests rapides):

`python -m src.pipeline.run_full_pipeline --max-posts-per-subreddit 100`

Important:
- Vérifier que `KAGGLE_USERNAME` et `KAGGLE_KEY` sont définis dans `.env`.
- Vérifier que `HEADERS` (ou `USER_AGENT`) est défini dans `.env` pour Reddit.

Pour un lancement par modèle, il est également possible d'ouvrir chaque notebook ayant le nom du modèle voulu en ayant pris soin de télécharger au préalable les données mises dans un dossier data/cleaned.

### DistilBERT et MentalRoBERTa

Les deux modèles ont été volontairement exclus de la pipeline principale par leur coût et temps de calcul, il est fortement recommandé de juste les télécharger dans le dossier models du Drive fourni ou alors d'exécuter individuellement les notebooks distilbert et mental_roberta sur Colab en ayant pris soin de télécharger en amont le dataset dans votre Google Drive et mis dans le bon dossier ou de changer le path lors de la lecture du csv.

!!! Il faut avoir au préalable demandé l'accès aux modèles sur HuggingFace avant de lancer les notebooks. !!!

### Lancement avec Docker
- Lancer les services: `docker compose up --build`

## Tests

Le projet contient une suite complète de tests unitaires et d'intégration avec **90+ cas de test**.

### Installation des dépendances de test
```bash
pip install pytest pytest-cov pytest-mock
```

### Exécuter les tests

```bash
# Tous les tests
pytest tests
```

Spécifiez le dossier pour des tests ciblés.

### Organisation des tests

**API** (`tests/api/test_api.py` - ~40 tests):
- Health check et endpoints disponibles
- Prédictions pour tous les modèles (lr, xgboost, distilbert, mental_roberta)
- Explications par mot avec HTML coloré et scores d'importance
- Statistiques agrégées et validation des compteurs
- Gestion d'erreurs (texte vide, modèle invalide, JSON malformé)
- Logging en arrière-plan et logging en base de données

**Dashboard** (`tests/dashboard/test_dashboard.py` - ~50+ tests):
- Fonctions utilitaires (formatage des niveaux de risque, bandes de probabilité)
- Traduction et support multilingue
- Intégration API (structure des requêtes/réponses)
- Formatage HTML et gestion d'état Streamlit
- Gestion des erreurs (timeouts, connexions, validation)
- Flux complets bout-en-bout (prédiction → explication → statistiques)

**Data Cleaning**:
- Téléchargement, nettoyage et équilibrage des données

**Training**
- Prétraitement, entraînement, évaluation et prédiction

**Pipeline**
- Orchestration de la pipeline complète

## Monitoring Cloud Run: alerte langues hors FR/EN

Une alerte a été conçue pour tester les notifications par mail lorsqu'au moins 20 requêtes ont été faites en moins de 20 minutes sur une langue autre que français et anglais.

Le backend émet désormais un log structuré par requête `/predict` avec:
- `event="predict_request"`
- `client_origin` (ex: `streamlit`)
- `source_language` (code détecté, ex: `es`, `fr-fr`, `en-us`)
- `is_non_fr_en` (`true` si la langue n'est ni française ni anglaise, variantes régionales incluses)

Des fichiers prêts à l'emploi sont fournis:
- `ops/monitoring/non_fr_en_streamlit_metric.yaml`
- `ops/monitoring/non_fr_en_streamlit_alert_policy.json`

### 1. Créer (ou mettre à jour) la métrique log-based

```bash
PROJECT_ID="tokyo-bedrock-483613-j3"

gcloud logging metrics update non_fr_en_streamlit_predict_count \
	--project "$PROJECT_ID" \
	--description "Count of Streamlit-origin /predict requests where detected source language is not French or English." \
	--log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="mental-health-api" AND jsonPayload.event="predict_request" AND jsonPayload.client_origin="streamlit" AND jsonPayload.is_non_fr_en=true' \
||
gcloud logging metrics create non_fr_en_streamlit_predict_count \
	--project "$PROJECT_ID" \
	--description "Count of Streamlit-origin /predict requests where detected source language is not French or English." \
	--log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="mental-health-api" AND jsonPayload.event="predict_request" AND jsonPayload.client_origin="streamlit" AND jsonPayload.is_non_fr_en=true'
```

### 2. Créer la policy d'alerte (seuil: 20 événements sur 20 minutes)

Le fichier `ops/monitoring/non_fr_en_streamlit_alert_policy.json` est configuré pour déclencher quand le compteur dépasse 19 (équivalent opérationnel à `>= 20`) sur une fenêtre de 1200 secondes.

```bash
PROJECT_ID="tokyo-bedrock-483613-j3"

gcloud alpha monitoring policies create \
	--project "$PROJECT_ID" \
	--policy-from-file ops/monitoring/non_fr_en_streamlit_alert_policy.json
```

### 3. Ajouter un canal de notification

La policy est créée avec `notificationChannels: []`.
Ajoutez ensuite votre canal (email, Slack webhook, PagerDuty...) dans le fichier JSON puis exécutez une mise à jour:

```bash
PROJECT_ID="tokyo-bedrock-483613-j3"
POLICY_ID="YOUR_POLICY_ID"

gcloud alpha monitoring policies update "$POLICY_ID" \
	--project "$PROJECT_ID" \
	--policy-from-file ops/monitoring/non_fr_en_streamlit_alert_policy.json
```
