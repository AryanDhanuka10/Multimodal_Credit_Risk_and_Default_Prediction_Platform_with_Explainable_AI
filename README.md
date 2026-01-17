# 🧠 Multimodal Credit Risk & Default Prediction Platform  
**Production-Ready, Explainable, End-to-End AI System**

> An end-to-end, multimodal AI platform that predicts customer credit risk by fusing **tabular financial data, transaction time-series, document images, and customer complaint text**, with full explainability and a deployable FastAPI backend.

---

## 🚀 Why This Project Exists

Traditional credit risk systems rely heavily on **tabular data alone**, ignoring:
- Transaction behavior patterns  
- Uploaded financial documents  
- Customer complaint narratives  

This project demonstrates how **modern AI systems** combine:
- Classical ML (LightGBM)
- Deep Learning (CNNs, embeddings)
- NLP (Transformers + topic modeling)
- System engineering (pipelines, testing, APIs)

to build **real, production-ready decision systems**.

---

## 🧩 Key Capabilities

- ✅ **Multimodal Risk Modeling**
  - Tabular credit features
  - Transaction time-series behavior
  - Document image embeddings (CNN)
  - Customer complaint text (NLP)

- ✅ **Hybrid AI Architecture**
  - ML + DL + NLP + rule-aware aggregation
  - Each modality contributes with confidence-weighted signals

- ✅ **Explainability Built-In**
  - Feature-level explanations (SHAP for tabular)
  - Modality-level contribution breakdown
  - Transparent final risk score

- ✅ **Production-Grade Backend**
  - FastAPI inference service
  - Dependency-injected inference engine
  - Fully testable without trained models

- ✅ **Engineering Best Practices**
  - Modular pipeline design
  - Pytest test suite
  - Clean package structure
  - Config-driven execution
  - No hard-coded paths or hacks

---

## 🏗️ System Architecture (High Level)

```

```
           ┌────────────────────┐
           │  Client / Frontend  │
           └─────────┬──────────┘
                     │
              FastAPI Backend
                     │
```

┌───────────────┬───────────────┬───────────────┬───────────────┐
│ Tabular ML    │ Time-Series   │ Vision Model  │ NLP Pipeline  │
│ (LightGBM)    │ Features      │ (CNN Embed)   │ (Topics)      │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┘
│               │               │               │
└───────────────┴───────────────┴───────────────┘
│
Confidence-Weighted Risk Aggregator
│
Final Risk Score + Explanation

````

---

## 🧠 Modeling Approach (Hybrid AI)

| Modality | Technique | Purpose |
|--------|----------|--------|
| Tabular | LightGBM | Core credit default prediction |
| Time-Series | Rolling statistical features | Detect abnormal spending behavior |
| Documents | CNN (ResNet embeddings) | Capture latent document risk signals |
| Text | Transformer embeddings + topic modeling | Extract risk-related complaint themes |
| Aggregation | Confidence-weighted fusion | Robust final decision |

> **Why not a single LLM?**  
> LLMs cannot reliably handle numerical precision, temporal patterns, or calibrated risk scoring. This system uses the *right model for the right signal*.

---

## 🔍 Explainability

The system provides:
- **Final Risk Score** (0–1)
- **Per-Modality Breakdown**
  - Score
  - Confidence
  - Weighted contribution
  - Percent impact on final decision

This makes the system suitable for **regulated domains** like finance.

---

## 🌐 API Endpoints

### Health Check
```http
GET /health
````

Response:

```json
{ "status": "ok" }
```

---

### Predict Credit Risk

```http
POST /predict
```

Example payload:

```json
{
  "tabular": {
    "features": {
      "f0": 0.2,
      "f1": 0.8
    }
  },
  "timeseries": {
    "values": [[0.1, 0.3, 0.2]]
  }
}
```

Response:

```json
{
  "final_risk_score": 0.47,
  "breakdown": {
    "tabular": { "score": 0.5, "confidence": 0.8 },
    "timeseries": { "score": 0.4, "confidence": 0.6 }
  }
}
```

---

## 🧪 Testing

* Unit tests for:

  * Feature engineering
  * Risk aggregation
  * Adapters
* Integration tests for:

  * Inference pipeline
  * FastAPI endpoints
* API tests **do not require trained models**

Run all tests:

```bash
pytest -v
```

---

## 🛠️ Tech Stack

* **Python 3.10**
* **FastAPI** – inference service
* **LightGBM** – tabular ML
* **PyTorch / Torchvision** – vision embeddings
* **Transformers / NLP** – text embeddings & topics
* **SHAP** – explainability
* **Pytest** – testing
* **Pandas / NumPy** – data handling

---

## 📁 Project Structure

```
src/Credit_Risk_Modelling/
├── api/                # FastAPI service
├── pipeline/           # Training & inference pipelines
├── components/         # Models, adapters, explainability
├── entity/             # Typed data contracts
├── config/             # Configuration management
└── constants/
```

---

## 🧠 What This Project Demonstrates

* End-to-end AI system thinking
* Real-world ML engineering practices
* Production-ready backend design
* Explainable AI for high-risk domains
* Clean, testable, modular codebase

This is **not a Kaggle project**.
This is a **deployable AI product**.

---

## 👤 Author

**Aryan**

B.Tech | AI / ML Engineer

Focused on building **production-grade AI systems**, not demos.

