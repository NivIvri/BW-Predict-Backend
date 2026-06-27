# BW-Predict — Backend

Node.js + Express API server for the birth weight prediction clinical tool.  
Handles authentication, patient/prediction database operations, and runs the XGBoost ML model via a Python subprocess.

> For a full deep-dive into how the project is built, see the [Frontend PROJECT_OVERVIEW.md](https://github.com/NivIvri/BW-Predict-Frontend/blob/main/PROJECT_OVERVIEW.md).

---

## Tech Stack

- Node.js + Express
- Supabase (PostgreSQL database + Auth verification)
- Python + XGBoost + SHAP (ML model)

---

## Setup

```bash
npm install
pip install joblib xgboost scikit-learn shap pandas numpy
```

Create a `.env` file:
```
SUPABASE_URL=https://ijrkemmsdhzbzuudrwej.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
PORT=5000
```

```bash
node server.js
# runs on http://localhost:5000
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/patients/lookup` | Find patient by ID number |
| POST | `/predict` | Run ML model + save prediction |
| GET | `/predictions` | Get all predictions for logged-in doctor |
| PATCH | `/predictions/:id/actual-weight` | Record actual birth weight after delivery |
| DELETE | `/predictions` | Clear all predictions for the doctor |

All routes except `/` require a `Authorization: Bearer <token>` header (Supabase JWT).

---

## ML Model

The model lives in `ml_model/`:
- `birth_weight_xgboost_model.joblib` — trained XGBoost model (30 features)
- `predict.py` — called as a subprocess, reads JSON from stdin, outputs prediction + SHAP values to stdout

Model accuracy: MAE ~118g, R² ~0.90

---

## Related

- **Frontend repo:** [BW-Predict-Frontend](https://github.com/NivIvri/BW-Predict-Frontend)
