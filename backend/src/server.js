require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const Minio = require('minio');

const app = express();
const PORT = process.env.PORT || 5000;
const MINIO_BUCKET = process.env.MINIO_BUCKET_NAME || 'resume-analyzer-bucket';

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

// Configure MinIO/S3-compatible client if environment variables provided
let minioClient = null;
const MINIO_ENDPOINT = process.env.MINIO_ENDPOINT;
if (MINIO_ENDPOINT && process.env.MINIO_ACCESS_KEY && process.env.MINIO_SECRET_KEY) {
  try {
    minioClient = new Minio.Client({
      endPoint: MINIO_ENDPOINT,
      port: Number(process.env.MINIO_PORT) || 9000,
      useSSL: process.env.MINIO_USE_SSL === 'true',
      accessKey: process.env.MINIO_ACCESS_KEY,
      secretKey: process.env.MINIO_SECRET_KEY,
    });
    console.log(`[INFO] MinIO client configured: ${MINIO_ENDPOINT}:${process.env.MINIO_PORT}`);
    console.log(`[INFO] Bucket: ${MINIO_BUCKET}`);
  } catch (err) {
    console.warn('[WARN] Failed to configure MinIO client:', err.message || err);
    minioClient = null;
  }
}

async function uploadToMinio(bucket, objectName, buffer) {
  if (!minioClient) return { uploaded: false, error: 'MinIO not configured' };

  try {
    // Ensure bucket exists (creates if missing)
    const exists = await minioClient.bucketExists(bucket).catch(() => false);
    if (!exists) {
      await minioClient.makeBucket(bucket, 'us-east-1');
    }

    await minioClient.putObject(bucket, objectName, buffer);
    return { uploaded: true, path: `minio://${bucket}/${objectName}` };
  } catch (err) {
    return { uploaded: false, error: err.message || String(err) };
  }
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

// Extract key information from resume
function extractResumeSignals(text) {
  const skills = [
    'python', 'javascript', 'java', 'nodejs', 'node.js', 'react', 'angular', 'vue',
    'sql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure', 'gcp', 'docker',
    'kubernetes', 'git', 'jenkins', 'devops', 'linux', 'html', 'css', 'rest',
    'microservices', 'api', 'agile', 'scrum', 'jira', 'kafka', 'redis', 'spring',
    'django', 'flask', 'express', 'fastapi', 'typescript', 'c++', 'golang', 'rust',
  ];

  const education = [
    'btech', 'b.tech', 'bachelor', 'master', 'mtech', 'm.tech', 'phd', 'certification',
    'diploma', 'degree', 'university', 'college', 'school',
  ];

  const actionVerbs = [
    'developed', 'built', 'created', 'designed', 'implemented', 'deployed',
    'managed', 'led', 'improved', 'optimized', 'automated', 'reduced', 'increased',
    'achieved', 'delivered', 'launched', 'architected', 'engineered', 'maintained',
  ];

  const lowerText = text.toLowerCase();

  // Extract skills
  const foundSkills = skills.filter((skill) => lowerText.includes(skill));

  // Extract education
  const foundEducation = education.filter((edu) => lowerText.includes(edu));

  // Find action verbs
  const foundActions = actionVerbs.filter((verb) => lowerText.includes(verb));

  // Find years of experience
  const yearsMatch = text.match(/(\d+)\s*\+?\s*years?/i);
  const yearsOfExp = yearsMatch ? parseInt(yearsMatch[1]) : 0;

  // Check for key sections
  const hasSummary = /summary|objective|profile/i.test(text);
  const hasExperience = /experience|employment|work|position/i.test(text);
  const hasProjects = /project|built|developed|created/i.test(text);
  const hasEducation = /education|bachelor|master|degree/i.test(text);
  const hasCertifications = /certification|certified|certificate/i.test(text);
  const hasMetrics = /\d+%|\d+\+?users?|\d+\+?customers?|increased|reduced/i.test(text);

  // Extract contact info
  const emailMatch = text.match(/\S+@\S+\.\S+/);
  const phoneMatch = text.match(/\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/);
  const linkedinMatch = text.match(/linkedin\.com\/in\/[\w-]+/i);

  return {
    foundSkills,
    foundEducation,
    foundActions,
    yearsOfExp,
    sections: {
      hasSummary,
      hasExperience,
      hasProjects,
      hasEducation,
      hasCertifications,
      hasMetrics,
    },
    contact: {
      email: emailMatch ? emailMatch[0] : null,
      phone: phoneMatch ? phoneMatch[0] : null,
      linkedin: linkedinMatch ? linkedinMatch[0] : null,
    },
  };
}

// Generate resume issues/mistakes
function generateResumeIssues(text, signals) {
  const issues = [];

  if (!signals.contact.email && !signals.contact.phone) {
    issues.push('Add contact information (email and phone) at the top of your resume.');
  }

  if (!signals.sections.hasSummary) {
    issues.push('Add a professional summary or objective section.');
  }

  if (!signals.sections.hasExperience) {
    issues.push('Ensure you have a clear Experience or Work History section.');
  }

  if (signals.foundSkills.length < 5) {
    issues.push('Add more technical skills - consider adding a dedicated Skills section.');
  }

  if (!signals.sections.hasProjects && signals.yearsOfExp > 0) {
    issues.push('Highlight specific projects or achievements you\'ve worked on.');
  }

  if (!signals.sections.hasMetrics) {
    issues.push('Include quantified achievements (numbers, percentages, impact metrics).');
  }

  if (text.split(/\s+/).length < 150) {
    issues.push('Resume seems too short - add more details about your experience and achievements.');
  }

  if (!signals.foundEducation.length) {
    issues.push('Include your educational background or certifications.');
  }

  if (signals.foundActions.length < 3) {
    issues.push('Use strong action verbs to describe your accomplishments (e.g., "built", "improved", "achieved").');
  }

  return issues.slice(0, 6); // Return top 6 issues
}

// Generate improvement plan
function generateImprovementPlan(text, signals, missingKeywords) {
  const plan = [];

  // Priority 1: Contact info
  if (!signals.contact.email || !signals.contact.phone) {
    plan.push('Step 1: Add your email and phone number prominently at the top.');
  }

  // Priority 2: Summary
  if (!signals.sections.hasSummary) {
    plan.push('Step 2: Create a brief professional summary (2-3 lines) highlighting your key expertise.');
  }

  // Priority 3: Experience
  if (!signals.sections.hasExperience) {
    plan.push('Step 3: Organize your work history with clear job titles, companies, and dates.');
  }

  // Priority 4: Achievements
  if (signals.foundActions.length < 5) {
    plan.push('Step 4: Rewrite experience bullets using action verbs (built, improved, increased, reduced).');
  }

  // Priority 5: Metrics
  if (!signals.sections.hasMetrics) {
    plan.push('Step 5: Quantify your impact (percentages, numbers, timelines, outcomes).');
  }

  // Priority 6: Skills match
  if (missingKeywords.length > 0) {
    const topMissing = missingKeywords.slice(0, 3).join(', ');
    plan.push(`Step 6: Add missing keywords from job description: ${topMissing}`);
  }

  return plan.slice(0, 6);
}

// Generate strength signals
function generateStrengthSignals(signals) {
  const strengths = [];

  if (signals.contact.email) strengths.push('✓ Email provided');
  if (signals.contact.phone) strengths.push('✓ Phone provided');
  if (signals.sections.hasSummary) strengths.push('✓ Professional summary');
  if (signals.sections.hasExperience) strengths.push('✓ Work experience');
  if (signals.sections.hasProjects) strengths.push('✓ Project descriptions');
  if (signals.foundSkills.length >= 5) strengths.push(`✓ ${signals.foundSkills.length} technical skills found`);
  if (signals.foundActions.length >= 5) strengths.push('✓ Strong action verbs used');
  if (signals.sections.hasMetrics) strengths.push('✓ Quantified achievements');
  if (signals.yearsOfExp > 0) strengths.push(`✓ ${signals.yearsOfExp}+ years experience`);
  if (signals.sections.hasEducation) strengths.push('✓ Education included');

  return strengths.slice(0, 8);
}

// Generate role recommendations
function generateRoleRecommendations(signals, jobFamily) {
  const roles = [];

  // Map skills to job families
  const techSkills = ['python', 'java', 'nodejs', 'react', 'angular', 'sql', 'api', 'rest'];
  const dataSkills = ['python', 'sql', 'tableau', 'powerbi', 'analytics', 'bi'];
  const cloudSkills = ['aws', 'azure', 'docker', 'kubernetes', 'devops', 'jenkins'];
  const webSkills = ['javascript', 'react', 'angular', 'html', 'css', 'nodejs'];

  // Count skill matches
  const techMatch = signals.foundSkills.filter((s) => techSkills.includes(s)).length;
  const dataMatch = signals.foundSkills.filter((s) => dataSkills.includes(s)).length;
  const cloudMatch = signals.foundSkills.filter((s) => cloudSkills.includes(s)).length;
  const webMatch = signals.foundSkills.filter((s) => webSkills.includes(s)).length;

  // Software Engineer
  if (techMatch >= 3 || signals.sections.hasProjects) {
    roles.push({
      role: 'Software Engineer',
      confidence: Math.min(0.95, 0.5 + techMatch * 0.1),
      reason: [
        `Found ${techMatch} relevant technical skills`,
        signals.sections.hasProjects ? 'Project experience detected' : 'Consider highlighting projects',
      ],
      bestFor: ['software'],
    });
  }

  // Data Scientist
  if (dataMatch >= 2) {
    roles.push({
      role: 'Data Scientist',
      confidence: Math.min(0.9, 0.4 + dataMatch * 0.15),
      reason: [
        `Data skills detected: ${signals.foundSkills.filter((s) => dataSkills.includes(s)).join(', ')}`,
        signals.sections.hasMetrics ? 'Analytics mindset evident' : 'Add data-driven achievements',
      ],
      bestFor: ['data'],
    });
  }

  // Cloud Engineer
  if (cloudMatch >= 2) {
    roles.push({
      role: 'Cloud Engineer / DevOps',
      confidence: Math.min(0.9, 0.45 + cloudMatch * 0.15),
      reason: [
        `Cloud expertise: ${signals.foundSkills.filter((s) => cloudSkills.includes(s)).join(', ')}`,
        signals.sections.hasExperience ? 'Infrastructure experience' : 'Highlight infrastructure work',
      ],
      bestFor: ['cloud'],
    });
  }

  // Web Developer
  if (webMatch >= 3) {
    roles.push({
      role: 'Web Developer / Full Stack',
      confidence: Math.min(0.92, 0.5 + webMatch * 0.1),
      reason: [
        `Web technologies: ${signals.foundSkills.filter((s) => webSkills.includes(s)).join(', ')}`,
        'Frontend/backend skills present',
      ],
      bestFor: ['web'],
    });
  }

  // Fallback: General recommendation
  if (roles.length === 0) {
    roles.push({
      role: 'Software Developer',
      confidence: 0.65,
      reason: [
        'Based on your background and experience',
        'Highlight specific technical skills to improve match',
      ],
      bestFor: ['software'],
    });
  }

  return roles.slice(0, 4);
}

function analyzeResume(resumeText, jobDescription) {
  const resumeTokens = tokenize(resumeText);
  const jdTokens = tokenize(jobDescription);

  const resumeTokenSet = new Set(resumeTokens);
  const uniqueJdTokens = [...new Set(jdTokens)].filter((word) => word.length > 3);

  const matchedKeywords = uniqueJdTokens.filter((word) => resumeTokenSet.has(word));
  const missingKeywords = uniqueJdTokens.filter((word) => !resumeTokenSet.has(word));

  // Calculate match percentage
  const matchPercentage = uniqueJdTokens.length === 0
    ? 0
    : Math.round((matchedKeywords.length / uniqueJdTokens.length) * 100);

  // Calculate confidence/accuracy based on multiple factors
  const resumeLength = resumeText.split(/\s+/).filter((w) => w.length > 0).length;
  const hasEmail = /\S+@\S+\.\S+/.test(resumeText);
  const hasPhone = /\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/.test(resumeText);
  const hasProjects = /project|built|developed|created|implemented|deployed/i.test(resumeText);
  const hasMetrics = /\d+%|\d+\+?years|increased|reduced|saved/i.test(resumeText);

  // Accuracy is based on data quality, not just matching
  let accuracyScore = 50; // baseline
  if (hasEmail) accuracyScore += 5;
  if (hasPhone) accuracyScore += 5;
  if (resumeLength > 100) accuracyScore += 10;
  if (hasProjects) accuracyScore += 15;
  if (hasMetrics) accuracyScore += 15;

  // Cap at 100
  accuracyScore = Math.min(100, accuracyScore);

  // Final score combines match + accuracy
  const finalScore = Math.round((matchPercentage * 0.6) + (accuracyScore * 0.4));

  // Extract resume signals
  const signals = extractResumeSignals(resumeText);
  
  const recommendations = [];
  if (matchPercentage < 40) {
    recommendations.push('Improve alignment with job description by adding relevant technical skills.');
  }
  if (missingKeywords.length > 0) {
    recommendations.push('Include measurable projects and keywords: ' + missingKeywords.slice(0, 8).join(', ') + '.');
  }
  if (!/project|experience|internship/i.test(resumeText)) {
    recommendations.push('Add a dedicated Experience or Projects section to strengthen your resume.');
  }

  // Generate analysis sections
  const resumeIssues = generateResumeIssues(resumeText, signals);
  const improvementPlan = generateImprovementPlan(resumeText, signals, missingKeywords);
  const strengthSignals = generateStrengthSignals(signals);
  const roleRecommendations = generateRoleRecommendations(signals, null);
  const skillGaps = missingKeywords.length > 0 ? missingKeywords : (
    signals.foundSkills.length < 5 ? ['Add more technical skills', 'Highlight specialized expertise'] : []
  );

  return {
    overallScore: finalScore,
    matchPercentage,
    accuracyScore,
    matchedKeywords,
    missingKeywords: missingKeywords.slice(0, 20),
    recommendations,
    // New fields for frontend sections
    strengthSignals,
    skillGaps,
    resumeIssues,
    improvementPlan,
    roleRecommendations,
    // Extracted info
    extractedSkills: signals.foundSkills,
    extractedEducation: signals.foundEducation,
    yearsOfExperience: signals.yearsOfExp,
    // Metrics
    metrics: {
      matchScore: matchPercentage,
      dataQuality: accuracyScore,
      hasContact: hasEmail || hasPhone,
      hasProjects,
      hasMetrics,
      resumeLength: resumeTokens.length,
    },
  };
}

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'cloud-resume-analyzer-api',
    timestamp: new Date().toISOString(),
    cloud: {
      provider: minioClient ? 'minio' : 'memory',
      endpoint: MINIO_ENDPOINT || 'none',
      storageBucket: MINIO_BUCKET,
      minioConnected: !!minioClient,
    },
  });
});

app.post('/api/analyze', upload.single('resume'), async (req, res) => {
  try {
    const { jobDescription = '' } = req.body;

    if (!req.file) {
      return res.status(400).json({ error: 'Resume file is required.' });
    }

    const filename = req.file.originalname || 'resume.txt';
    const storagePath = `minio://${MINIO_BUCKET}/${Date.now()}-${filename}`;

    // For basic version, parse text directly from uploaded buffer.
    const resumeText = req.file.buffer.toString('utf-8');

    if (!resumeText.trim()) {
      return res.status(400).json({ error: 'Uploaded resume appears to be empty or unsupported.' });
    }

    const analysis = analyzeResume(resumeText, jobDescription);

    // Attempt to upload to MinIO if configured
    let cloudUpload = { uploaded: false, path: storagePath };
    if (minioClient) {
      const objectName = `${Date.now()}-${filename}`;
      const result = await uploadToMinio(MINIO_BUCKET, objectName, req.file.buffer);
      if (result.uploaded) {
        cloudUpload = { uploaded: true, path: result.path };
      } else {
        cloudUpload = { uploaded: false, error: result.error, path: storagePath };
      }
    }

    return res.json({
      message: 'Resume analyzed successfully.',
      file: {
        name: filename,
        sizeBytes: req.file.size,
        mimeType: req.file.mimetype,
      },
      cloudStorage: cloudUpload,
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
