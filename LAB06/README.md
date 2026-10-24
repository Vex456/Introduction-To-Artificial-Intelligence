# LAB 06 — Diabetes Prediction Web Application

This lab deploys the trained scikit-learn Logistic Regression model (`My_Model.pkl`) from Lab 4 as an interactive web service using **FastAPI** and a lightweight web frontend.

---

## 1. Project Files

- `main.py`: FastAPI backend REST API serving `/predict` and `/health` endpoints.
- `index.html`: Client-side interface to input patient parameters and view predictions in real-time.
- `My_Model.pkl`: Pre-trained logistic regression classifier serialized with `pickle`.

---

## 2. Prerequisites

Ensure required packages are installed:

```bash
pip install fastapi uvicorn pandas scikit-learn
```

---

## 3. Running the Application

### Step 1: Start the FastAPI Backend
From the `LAB06` directory, start the Uvicorn ASGI server:

```bash
python -m uvicorn main:app --reload --port 8000
```

- API Base: [http://localhost:8000/](http://localhost:8000/)
- Interactive Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Step 2: Launch the Frontend
Open `index.html` in any web browser (double-click or serve via Live Server).

Enter patient data:
- **Glucose** (e.g., `112`)
- **Blood Pressure** (e.g., `75`)
- **Insulin** (e.g., `102.5`)
- **BMI** (e.g., `35.7`)
- **Diabetes Pedigree Function** (e.g., `0.148`)
- **Age** (e.g., `21`)

Click **Predict** to receive the classification result (`Diabetic` or `Not Diabetic`).
