const express = require('express');
const cors = require('cors');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function analyzeResume(resumeText, jobDescription) {
  const resumeTokens = tokenize(resumeText);
  const jdTokens = tokenize(jobDescription);

  const resumeTokenSet = new Set(resumeTokens);
  const uniqueJdTokens = [...new Set(jdTokens)].filter((word) => word.length > 3);

  const matchedKeywords = uniqueJdTokens.filter((word) => resumeTokenSet.has(word));
  const missingKeywords = uniqueJdTokens.filter((word) => !resumeTokenSet.has(word));

  const score = uniqueJdTokens.length === 0
    ? 0
    : Math.round((matchedKeywords.length / uniqueJdTokens.length) * 100);

  const recommendations = [];
  if (score < 40) {
    recommendations.push('Improve alignment with job description by adding relevant technical skills.');
  }
  if (missingKeywords.length > 0) {
    recommendations.push('Include measurable projects and keywords: ' + missingKeywords.slice(0, 8).join(', ') + '.');
  }
  if (!/project|experience|internship/i.test(resumeText)) {
    recommendations.push('Add a dedicated Experience or Projects section to strengthen your resume.');
  }

  return {
    overallScore: score,
    matchedKeywords,
    missingKeywords: missingKeywords.slice(0, 20),
    recommendations,
  };
}

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'cloud-resume-analyzer-api',
    timestamp: new Date().toISOString(),
    cloud: {
      provider: 'demo-cloud',
      region: 'us-east-1',
      storageBucket: 'resume-analyzer-bucket',
    },
  });
});

app.post('/api/analyze', upload.single('resume'), (req, res) => {
  try {
    const { jobDescription = '' } = req.body;

    if (!req.file) {
      return res.status(400).json({ error: 'Resume file is required.' });
    }

    const filename = req.file.originalname || 'resume.txt';
    const storagePath = `cloud://resume-analyzer-bucket/${Date.now()}-${filename}`;

    // For basic version, parse text directly from uploaded buffer.
    const resumeText = req.file.buffer.toString('utf-8');

    if (!resumeText.trim()) {
      return res.status(400).json({ error: 'Uploaded resume appears to be empty or unsupported.' });
    }

    const analysis = analyzeResume(resumeText, jobDescription);

    return res.json({
      message: 'Resume analyzed successfully.',
      file: {
        name: filename,
        sizeBytes: req.file.size,
        mimeType: req.file.mimetype,
      },
      cloudStorage: {
        uploaded: true,
        path: storagePath,
      },
      analysis,
    });
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to analyze resume.',
      details: error.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
