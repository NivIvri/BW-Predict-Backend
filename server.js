require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const app = express();
app.use(cors());
app.use(express.json());

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const PORT = process.env.PORT || 5000;

// Service-role client — bypasses RLS, used for all DB operations
const db = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Anon client — used only to verify the user's JWT token
const authClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Auth middleware — verifies the Bearer token and attaches the user
async function requireAuth(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  const { data: { user }, error } = await authClient.auth.getUser(token);
  if (error || !user) return res.status(401).json({ error: 'Invalid token' });

  req.user = user;
  next();
}

const ML_SCRIPT = path.join(__dirname, 'ml_model', 'predict.py');

function runModel(payload) {
  return new Promise((resolve, reject) => {
    const py = spawn('python', [ML_SCRIPT]);
    let stdout = '';
    let stderr = '';
    py.stdout.on('data', chunk => { stdout += chunk; });
    py.stderr.on('data', chunk => { stderr += chunk; });
    py.on('close', code => {
      if (code !== 0) return reject(new Error(`ML model exited ${code}: ${stderr}`));
      try {
        resolve(JSON.parse(stdout));
      } catch (e) {
        reject(new Error(`Failed to parse model output: ${stdout}`));
      }
    });
    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();
  });
}

// ── Health check ────────────────────────────────────────────────────
app.get('/', (req, res) => res.send('SERVER IS RUNNING'));

// ── Patient lookup ──────────────────────────────────────────────────
app.post('/patients/lookup', requireAuth, async (req, res) => {
  const { idNumber } = req.body;
  const { data } = await db
    .from('patients')
    .select('*')
    .eq('id_number', idNumber)
    .maybeSingle();

  res.json({ patient: data ?? null });
});

// ── Get all predictions for the logged-in doctor ────────────────────
app.get('/predictions', requireAuth, async (req, res) => {
  const { data, error } = await db
    .from('predictions')
    .select('*, patients(*)')
    .eq('doctor_id', req.user.id)
    .order('created_at', { ascending: false });

  if (error) return res.status(500).json({ error: error.message });
  res.json({ predictions: data ?? [] });
});

// ── Submit prediction (ML + save to DB) ────────────────────────────
app.post('/predict', requireAuth, async (req, res) => {
  const { maternalInfo, obstetricHistory, fetalBiometry, patientInfo } = req.body;

  try {
    // 1. Get or create patient
    const { data: existing } = await db
      .from('patients')
      .select('*')
      .eq('id_number', patientInfo.idNumber)
      .maybeSingle();

    let patient = existing;
    if (!patient) {
      const { data: created, error: createErr } = await db
        .from('patients')
        .insert({
          id_number: patientInfo.idNumber,
          name: patientInfo.name,
          family_name: patientInfo.familyName,
          height: Number(maternalInfo.height) || null,
        })
        .select()
        .single();
      if (createErr) throw new Error(createErr.message);
      patient = created;
    }

    // 2. Run ML model via Python subprocess
    const { predictedWeight, shapExplanation } = await runModel({ maternalInfo, obstetricHistory, fetalBiometry });

    // 3. (SHAP explanation returned by runModel above)

    // 4. Save prediction to DB
    const { data: saved, error: saveErr } = await db
      .from('predictions')
      .insert({
        patient_id:                  patient.id,
        doctor_id:                   req.user.id,
        age:                         Number(maternalInfo.age) || null,
        pre_pregnancy_weight:        Number(maternalInfo.weightBeforePregnancy) || null,
        current_weight:              Number(maternalInfo.currentWeight) || null,
        bmi:                         Number(maternalInfo.bmi) || null,
        gestational_age_days:        Number(fetalBiometry.gestationalAge) * 7 || null,
        smoking:                     maternalInfo.smoking,
        alcohol:                     maternalInfo.alcohol,
        drugs:                       maternalInfo.drugs,
        gdm:                         maternalInfo.gdm,
        dm:                          maternalInfo.dm,
        g_count:                     Number(obstetricHistory.gravida) || null,
        p_count:                     Number(obstetricHistory.para) || null,
        ab_count:                    Number(obstetricHistory.abortions) || null,
        cs_count:                    Number(obstetricHistory.cesareanSections) || null,
        lc_count:                    Number(obstetricHistory.livingChildren) || null,
        eup_count:                   Number(obstetricHistory.eup) || null,
        vbac_count:                  Number(obstetricHistory.vbac) || null,
        past_births_average_weight:  Number(obstetricHistory.pastBirthsAverageWeight) || null,
        fetal_sex:                   fetalBiometry.fetalSex ?? null,
        sonographic_weight_estimate: Number(fetalBiometry.efwUltrasound) || null,
        clinical_weight_estimate:    Number(fetalBiometry.clinicalEstimation) || null,
        predicted_birth_weight:      predictedWeight,
        shap_explanation:            shapExplanation,
      })
      .select('*, patients(*)')
      .single();

    if (saveErr) throw new Error(saveErr.message);
    res.json({ prediction: saved });
  } catch (err) {
    console.error('predict error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Update actual birth weight ──────────────────────────────────────
app.patch('/predictions/:id/actual-weight', requireAuth, async (req, res) => {
  const { id } = req.params;
  const { actualWeight, predictedWeight } = req.body;

  const deviation = Math.round(((predictedWeight - actualWeight) / actualWeight) * 1000) / 10;

  const { error } = await db
    .from('predictions')
    .update({
      actual_birth_weight: actualWeight,
      prediction_error:    deviation,
      updated_at:          new Date().toISOString(),
    })
    .eq('id', id);

  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true, deviation });
});

// ── Clear all predictions for the doctor ───────────────────────────
app.delete('/predictions', requireAuth, async (req, res) => {
  const { error } = await db
    .from('predictions')
    .delete()
    .eq('doctor_id', req.user.id);

  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true });
});

app.listen(PORT, () => console.log(`http://localhost:${PORT}`));
