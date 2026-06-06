require('dotenv').config();
const express = require('express');
const cors = require('cors');
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

    // 2. ML model placeholder — replace this block with the real model call
    const efwUltrasound   = Number(fetalBiometry?.efwUltrasound) || 3200;
    const clinicalEst     = Number(fetalBiometry?.clinicalEstimation) || 3200;
    const bmi             = Number(maternalInfo?.bmi) || 23;
    const bmiAdjust       = bmi > 30 ? 80 : bmi < 18.5 ? -60 : 0;
    const predictedWeight = Math.round((efwUltrasound * 0.6 + clinicalEst * 0.4) + bmiAdjust);

    // 3. SHAP explanation placeholder
    const shapExplanation = [
      { factor: 'Ultrasound EFW',        value: efwUltrasound,                        contribution: 320, direction: 'positive' },
      { factor: 'Clinical Estimation',   value: Number(fetalBiometry.clinicalEstimation) || 3100, contribution: 180, direction: 'positive' },
      { factor: 'Gestational Age',       value: Number(fetalBiometry.gestationalAge) || 38,        contribution: 140, direction: 'positive' },
      { factor: 'Abdominal Circumference', value: Number(fetalBiometry.ac) || 340,                contribution: 90,  direction: 'positive' },
      { factor: 'Maternal BMI',          value: bmi, contribution: bmi > 25 ? 60 : -40, direction: bmi > 25 ? 'positive' : 'negative' },
      { factor: 'AFI',                   value: Number(fetalBiometry.afi) || 12,                  contribution: 30,  direction: 'positive' },
    ].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    // 4. Save prediction to DB
    const { data: saved, error: saveErr } = await db
      .from('predictions')
      .insert({
        patient_id:                  patient.id,
        doctor_id:                   req.user.id,
        age:                         Number(maternalInfo.age) || null,
        pre_pregnancy_weight:        Number(maternalInfo.weightBeforePregnancy) || null,
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
        hc:                          Number(fetalBiometry.hc) || null,
        ac:                          Number(fetalBiometry.ac) || null,
        fl:                          Number(fetalBiometry.fl) || null,
        afi:                         Number(fetalBiometry.afi) || null,
        induction:                   fetalBiometry.induction,
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
