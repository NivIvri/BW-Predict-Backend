import sys
import json
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import shap

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'birth_weight_xgboost_model.joblib')

_obj = joblib.load(MODEL_PATH)
_model = _obj['model']
_features = _obj['features']
_explainer = shap.TreeExplainer(_model)

FEATURE_LABELS = {
    'gestational_age': 'Gestational Age',
    'smoking': 'Smoking',
    'alcohol': 'Alcohol',
    'drugs': 'Drug Use',
    'pregnancy_age': 'Maternal Age',
    'ab': 'Abortions',
    'cs': 'Cesarean Sections',
    'eup': 'EUP',
    'g': 'Gravida',
    'lc': 'Living Children',
    'p': 'Para',
    'vbac': 'VBAC',
    'dm_0': 'No Diabetes',
    'dm_1': 'Gestational DM',
    'dm_2': 'Pre-existing DM',
    'mother_height': 'Maternal Height',
    'pre_pregnancy_weight': 'Pre-Pregnancy Weight',
    'current_weight': 'Current Weight',
    'maternal_weight_gain': 'Maternal Weight Gain',
    'bmi': 'Maternal BMI',
    'us_estimated_weight': 'Ultrasound EFW',
    'clinical_estimated_weight': 'Clinical Estimation',
    'fetal_sex': 'Fetal Sex',
    'avg_previous_birth_weights': 'Avg Previous Birth Weight',
}


def _fval(v):
    try:
        return float(v) if v is not None and v != '' else None
    except (ValueError, TypeError):
        return None


def _bval(v):
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, str):
        return 1 if v.lower() in ('true', '1', 'yes') else 0
    return 1 if v else 0


def predict(data):
    m = data.get('maternalInfo', {})
    o = data.get('obstetricHistory', {})
    f = data.get('fetalBiometry', {})

    height = _fval(m.get('height'))
    pre_weight = _fval(m.get('weightBeforePregnancy'))
    curr_weight = _fval(m.get('currentWeight'))
    bmi = _fval(m.get('bmi'))
    avg_prev = _fval(o.get('pastBirthsAverageWeight'))

    weight_gain = (curr_weight - pre_weight) if (curr_weight is not None and pre_weight is not None) else None

    gdm = _bval(m.get('gdm'))
    dm = _bval(m.get('dm'))
    dm_0 = 1 if (not gdm and not dm) else 0
    dm_1 = 1 if gdm else 0
    dm_2 = 1 if dm else 0

    row = {
        'gestational_age':               _fval(f.get('gestationalAge')) or 0,
        'smoking':                        _bval(m.get('smoking')),
        'alcohol':                        _bval(m.get('alcohol')),
        'drugs':                          _bval(m.get('drugs')),
        'pregnancy_age':                  _fval(m.get('age')) or 0,
        'ab':                             _fval(o.get('abortions')) or 0,
        'cs':                             _fval(o.get('cesareanSections')) or 0,
        'eup':                            _fval(o.get('eup')) or 0,
        'g':                              _fval(o.get('gravida')) or 0,
        'lc':                             _fval(o.get('livingChildren')) or 0,
        'p':                              _fval(o.get('para')) or 0,
        'vbac':                           _fval(o.get('vbac')) or 0,
        'dm_0':                           dm_0,
        'dm_1':                           dm_1,
        'dm_2':                           dm_2,
        'mother_height':                  height if height is not None else 0,
        'height_is_missing':              0 if height is not None else 1,
        'pre_pregnancy_weight':           pre_weight if pre_weight is not None else 0,
        'pre_weight_is_missing':          0 if pre_weight is not None else 1,
        'current_weight':                 curr_weight if curr_weight is not None else 0,
        'curr_weight_is_missing':         0 if curr_weight is not None else 1,
        'maternal_weight_gain':           weight_gain if weight_gain is not None else 0,
        'maternal_weight_gain_is_missing': 0 if weight_gain is not None else 1,
        'bmi':                            bmi if bmi is not None else 0,
        'bmi_is_missing':                 0 if bmi is not None else 1,
        'us_estimated_weight':            _fval(f.get('efwUltrasound')) or 0,
        'clinical_estimated_weight':      _fval(f.get('clinicalEstimation')) or 0,
        'fetal_sex':                      _fval(f.get('fetalSex')) or 0,
        'avg_previous_birth_weights':     avg_prev if avg_prev is not None else 0,
        'avg_previous_is_missing':        0 if avg_prev is not None else 1,
    }

    df = pd.DataFrame([row], columns=_features)
    predicted_weight = round(float(_model.predict(df)[0]))

    shap_vals = _explainer.shap_values(df)
    shap_arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

    explanation = []
    for i, feat in enumerate(_features):
        if feat.endswith('_is_missing'):
            continue
        contrib = float(shap_arr[i])
        if abs(contrib) < 0.5:
            continue
        explanation.append({
            'factor': FEATURE_LABELS.get(feat, feat),
            'value': row[feat],
            'contribution': round(contrib, 1),
            'direction': 'positive' if contrib > 0 else 'negative',
        })
    explanation.sort(key=lambda x: abs(x['contribution']), reverse=True)

    return {'predictedWeight': predicted_weight, 'shapExplanation': explanation}


if __name__ == '__main__':
    input_data = json.loads(sys.stdin.read())
    result = predict(input_data)
    print(json.dumps(result))
