# Cloud Based Resume Analyzer (Mini Project)

A basic cloud-computing themed mini project with:
- Frontend: HTML, CSS, JavaScript
- Backend: Node.js + Express
- Feature: Upload resume + compare with job description + get a score and recommendations

## Project Structure

cloud-resume-analyzer/
- backend/
  - src/server.js
  - package.json
- frontend/
  - index.html
  - styles.css
  - script.js

## 1) Start Backend

```bash
cd backend
npm install
npm run dev
```

Backend runs at: `http://localhost:5000`

## 2) Run Frontend

Open `frontend/index.html` in your browser.

For better local development, use VS Code Live Server extension or any static server.

## API Endpoints

- `GET /api/health` : service health and mock cloud metadata
- `POST /api/analyze` : upload file + analyze against job description
  - `multipart/form-data`
  - field `resume`: file upload
  - field `jobDescription`: text

## Notes

- Basic version reads uploaded file as plain text (`.txt` works best).
- You can extend this by adding PDF/DOCX parsing and real cloud storage (AWS S3, Azure Blob, or GCP Storage).

## Suggested Next Upgrades

1. Add authentication (JWT / OAuth).
2. Store analysis history in a cloud database.
3. Add real ATS-style scoring categories.
4. Deploy backend on a cloud service (Azure App Service / AWS Elastic Beanstalk / Render).
