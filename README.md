# Certification-Thomas-DINH

## Run With Docker (Split Services)

This project is set up to run as two containers:
- API (FastAPI) on port 8000
- Dashboard (Streamlit) on port 8501

### 1) Configure environment

Copy .env-example to .env and set your values.

Important variables:
- API_URL and API_URL_LOCAL
- DATABASE_URL
- CORS_ALLOWED_ORIGINS
- LOCATION, PROJECT_ID, REPOSITORY, IMAGE

### 2) Build images locally

```bash
docker build -t mental-health-api:local -f Dockerfile .
docker build -t mental-health-dashboard:local -f Dockerfile.streamlit .
```

### 3) Run locally

```bash
docker run --rm -p 8000:8000 --env-file .env mental-health-api:local
docker run --rm -p 8501:8501 --env-file .env -e API_URL=http://host.docker.internal:8000 -e API_URL_LOCAL=http://host.docker.internal:8000 mental-health-dashboard:local
```

Open Streamlit at http://localhost:8501.

### 4) Optional local compose

```bash
docker compose up --build
```

## Deploy to Cloud Run

### 1) Build and push to Artifact Registry

```bash
export LOCATION=europe-west1
export PROJECT_ID=your-gcp-project-id
export REPOSITORY=your-artifact-registry-repository

export API_IMAGE=${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/mental-health-api:latest
export DASHBOARD_IMAGE=${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/mental-health-dashboard:latest

gcloud auth configure-docker ${LOCATION}-docker.pkg.dev

docker build -t ${API_IMAGE} -f Dockerfile .
docker build -t ${DASHBOARD_IMAGE} -f Dockerfile.streamlit .

docker push ${API_IMAGE}
docker push ${DASHBOARD_IMAGE}
```

### 2) Deploy API service

```bash
gcloud run deploy mental-health-api \
	--image ${API_IMAGE} \
	--region ${LOCATION} \
	--project ${PROJECT_ID} \
	--allow-unauthenticated \
	--set-env-vars DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DB_NAME',CORS_ALLOWED_ORIGINS='https://YOUR_DASHBOARD_URL'
```

Save the returned URL as API_SERVICE_URL.

### 3) Deploy Streamlit service

```bash
gcloud run deploy mental-health-dashboard \
	--image ${DASHBOARD_IMAGE} \
	--region ${LOCATION} \
	--project ${PROJECT_ID} \
	--allow-unauthenticated \
	--set-env-vars API_URL='https://API_SERVICE_URL',API_URL_LOCAL='https://API_SERVICE_URL'
```

## Notes

- Keep CORS_ALLOWED_ORIGINS=* only for quick tests, then lock it to the Streamlit URL.
- Use PostgreSQL in production so stats persist across restarts.
- Existing lowercase dockerfile is kept for compatibility; Dockerfile and Dockerfile.streamlit are the primary build files.
