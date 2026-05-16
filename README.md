# Cloud-Based Resume Analyzer

An intelligent resume analysis system that uses AI/ML and cloud storage to help candidates understand their career fit. Upload a resume, get instant analysis, role recommendations, and improvement suggestions with persistent storage on MinIO.

## Overview

- **Frontend**: React-inspired responsive UI (HTML, CSS, JavaScript)
- **Backend**: Express.js REST API with MinIO cloud storage integration
- **ML Service**: Flask-based NLP analyzer with sentence-transformers for semantic matching
- **Storage**: MinIO (S3-compatible) for resume persistence
- **Features**:
  - Resume upload with automatic text extraction
  - Job description matching (TF-IDF similarity)
  - Role recommendation engine (Data, Cloud, Web, Mobile, Finance, etc.)
  - Skill gap analysis and improvement planning
  - Resume quality scoring
  - Analysis history tracking

## Architecture

```
cloud-resume-analyzer/
├── backend/
│   ├── src/server.js              # Express API + MinIO integration
│   ├── package.json               # Node.js dependencies
│   ├── .env                       # MinIO config (create from .env.example)
│   ├── .env.example               # Template with default values
│   ├── mlService/
│   │   ├── app.py                 # Flask ML analyzer service
│   │   ├── requirements.txt       # Python dependencies
│   │   └── models/                # Trained model files
│   └── uploads/                   # Local upload cache
├── frontend/
│   ├── index.html                 # Main UI
│   ├── script.js                  # Frontend logic
│   ├── styles.css                 # Styling
│   └── package.json               # Frontend scripts
└── README.md
```

## Prerequisites

- **Node.js** 18+ (for backend)
- **Python** 3.9+ (for ML service)
- **MinIO** running on Windows (or accessible via network)
- **npm** and **pip** package managers

## Quick Start (All Services)

### 1. Configure MinIO Connection

Edit `backend/.env` with your MinIO credentials:

```env
MINIO_ENDPOINT=192.168.31.106
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_USE_SSL=false
MINIO_BUCKET_NAME=resume-analyzer-bucket
PORT=5000
```

**Windows MinIO Example** (from your setup):
```powershell
# MinIO is already running, so use:
MINIO_ENDPOINT=192.168.31.106:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### 2. Start Backend API

```powershell
cd backend
npm install
npm run dev
```

**Expected Output:**
```
[INFO] MinIO client configured: 192.168.31.106:9000
[INFO] Bucket: resume-analyzer-bucket
Server running on http://localhost:5000
```

### 3. Start Frontend Static Server

```powershell
cd frontend
npm run dev
# or: python -m http.server 5500
```

Open browser: **http://localhost:5500**

<<<<<<< HEAD
### 4. Start ML Service (Optional but Recommended)
=======
## 2.1) ML Service

A separate ML analyzer service lives in `backend/mlService`. Run it with Python if you want richer role prediction, skill extraction, and explainable resume analysis.

## 2.2) Blockchain Verification

A blockchain verification layer is available under `backend/blockchain`. It includes a `ResumeVerification` Solidity contract and deployment script for a local Hardhat network.

## API Endpoints
>>>>>>> dd0acc4 (updated code but with errors)

```powershell
cd backend/mlService
python -m venv .venv
& '.\.venv\Scripts\Activate.ps1'
pip install -r requirements.txt
python app.py
```

**Expected Output:**
```
[INFO] Starting ML Analyzer service on port 5001...
 * Running on http://0.0.0.0:5001
```

---

## API Reference

### Backend Endpoints

#### `GET /api/health`
Check backend and MinIO connection status.

**Response:**
```json
{
  "status": "ok",
  "service": "cloud-resume-analyzer-api",
  "cloud": {
    "provider": "minio",
    "endpoint": "192.168.31.106",
    "storageBucket": "resume-analyzer-bucket",
    "minioConnected": true
  }
}
```

#### `POST /api/analyze`
Upload resume and analyze against job description.

**Request:**
```
multipart/form-data
- resume: file (.txt, .pdf, .docx)
- jobDescription: string
```

**Response:**
```json
{
  "message": "Resume analyzed successfully.",
  "file": {
    "name": "resume.txt",
    "sizeBytes": 906,
    "mimeType": "text/plain"
  },
  "cloudStorage": {
    "uploaded": true,
    "path": "minio://resume-analyzer-bucket/1778003710154-resume.txt"
  },
  "analysis": {
    "overallScore": 100,
    "matchedKeywords": ["senior", "backend", "engineer"],
    "missingKeywords": [],
    "recommendations": []
  }
}
```

### ML Service Endpoints

#### `GET /health`
Check ML service status.

#### `POST /analyze`
Send resume text for ML-based analysis (role classification, semantic matching).

---

## Testing the System

### 1. Test Backend API

```powershell
# Health check
curl.exe http://localhost:5000/api/health

# Upload test resume
curl.exe -F "resume=@test_resume.txt" `
  -F "jobDescription=Backend engineer with Node.js" `
  http://localhost:5000/api/analyze
```

### 2. Test Frontend UI

1. Open **http://localhost:5500** in browser
2. Click on dropzone or select file
3. Paste job description (optional)
4. Click **Analyze**
5. View results and improvement plan

### 3. Verify Files in MinIO

**Option A: MinIO Web UI**
- Open **http://192.168.31.106:9001**
- Login: `minioadmin` / `minioadmin`
- Navigate to `resume-analyzer-bucket` bucket
- Confirm uploaded resumes are visible

**Option B: MinIO CLI (mc)**
```powershell
# Install mc from https://min.io/download#minio-client
mc alias set myminio http://192.168.31.106:9000 minioadmin minioadmin
mc ls myminio/resume-analyzer-bucket
mc cat myminio/resume-analyzer-bucket/<filename>
```

---

## Troubleshooting

### MinIO Connection Failed

**Error:** `Failed to configure MinIO client`

**Solutions:**
1. Verify MinIO is running: `http://192.168.31.106:9001`
2. Check `.env` file exists and has correct credentials
3. Confirm network can reach MinIO endpoint
4. Check firewall allows port 9000 (API) and 9001 (WebUI)

### Upload Endpoint Returns `uploaded: false`

**Reason:** MinIO client not initialized

**Fix:**
1. Restart backend after updating `.env`
2. Confirm `/api/health` shows `minioConnected: true`
3. Check backend console for `[INFO] MinIO client configured` message

### Frontend Cannot Reach Backend

**Error:** `Failed to analyze resume`

**Solutions:**
1. Verify backend is running: `curl.exe http://localhost:5000/api/health`
2. Check `script.js` has correct API_BASE_URL (`http://localhost:5000`)
3. Ensure CORS is enabled (default in `server.js`)

### ML Service Not Found

**Error:** `Cannot GET /analyze`

**Fix:**
- ML service is optional; frontend works without it
- To enable: start Flask service on port 5001 (see step 4 above)

---

## Environment Variables

Create `.env` in `backend/` directory:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MINIO_ENDPOINT` | `localhost` | MinIO server hostname/IP |
| `MINIO_PORT` | `9000` | MinIO API port |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO root user |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO root password |
| `MINIO_USE_SSL` | `false` | Enable TLS (for remote servers) |
| `MINIO_BUCKET_NAME` | `resume-analyzer-bucket` | S3 bucket name |
| `PORT` | `5000` | Backend server port |
| `NODE_ENV` | `development` | Node environment |

---

## File Upload Workflow

1. **Frontend**: User selects file via dropzone/input
2. **Frontend**: POST to `/api/analyze` with `multipart/form-data`
3. **Backend**: Receives and analyzes resume text
4. **Backend**: Uploads file buffer to MinIO bucket
5. **Backend**: Returns analysis + cloud storage metadata
6. **Frontend**: Displays score, recommendations, and history

**Cloud Storage Path Format:**
```
minio://resume-analyzer-bucket/{TIMESTAMP}-{FILENAME}
```

Example: `minio://resume-analyzer-bucket/1778003710154-resume.txt`

---

## Local Development

### Watch Mode (Backend)

Backend automatically reloads on file changes:
```powershell
npm run dev    # Uses nodemon
```

### Create Sample Resumes

```powershell
# Sample resume for testing
@"
Name: Jane Developer
Skills: Python, Docker, AWS
Experience: 3 years as Cloud Engineer
"@ | Set-Content sample_resume.txt

# Test upload
curl.exe -F "resume=@sample_resume.txt" http://localhost:5000/api/analyze
```

---

## Deployment Notes

### To Production

1. Use environment-specific `.env` files (`.env.production`)
2. Enable SSL: `MINIO_USE_SSL=true`
3. Use strong credentials (not `minioadmin/minioadmin`)
4. Deploy backend to cloud service (Render, Railway, Vercel)
5. Deploy frontend to CDN (GitHub Pages, Netlify, Vercel)
6. Use external MinIO instance or S3-compatible service

### Docker Support

(Optional) Create `Dockerfile` for containerized deployment:
```dockerfile
FROM node:18
WORKDIR /app
COPY backend .
RUN npm install
EXPOSE 5000
CMD ["npm", "run", "dev"]
```

---

## Next Steps / Future Enhancements

1. ✅ MinIO cloud storage integration
2. 📄 PDF/DOCX parsing (pypdf, python-docx)
3. 🔐 Authentication (JWT / OAuth)
4. 📊 Analysis history in database (MongoDB, PostgreSQL)
5. 🎯 Real ATS-style scoring categories
6. 🌐 Multi-language support
7. 📧 Email report export
8. 🔄 Batch resume analysis
9. 💼 Recruiter dashboard
10. 🚀 Kubernetes deployment

---

## Support

- **Backend Issues**: Check `backend/` console output
- **MinIO Issues**: Check MinIO WebUI at `http://192.168.31.106:9001`
- **Frontend Issues**: Check browser DevTools (F12 → Console)
- **ML Service Issues**: Check Python service logs

---

**Created for presentation on May 5, 2026**
