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
- `API_URL` et `API_URL_LOCAL` pour le dashboard.
- `GDRIVE_MODEL_FOLDER_ID` si vous voulez que l’API télécharge automatiquement les modèles manquants depuis Google Drive.
- `HEADERS` pour les scripts de collecte de données.

### Modèles
Le dépôt ne contient pas tous les artefacts de modèle. Pour un clone vierge, il faut soit:
- copier le dossier `models` complet à la racine du projet, avec les fichiers attendus par l’API,
- soit renseigner `GDRIVE_MODEL_FOLDER_ID` dans `.env` pour permettre le téléchargement automatique des modèles manquants au démarrage.

Les modèles sont disponibles ici: https://drive.google.com/drive/folders/14HIhHMygB9t0E66EMmORX2Bcd8SWAjME?hl=fr

### Lancement local
- Lancer l’API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- Lancer le dashboard: `streamlit run src/dashboard/app.py`

### Lancement avec Docker
- Lancer les services: `docker compose up --build`
