# Exposure-Aware Air Quality Health Risk Classification (ML)

## 📌 Overview
This project builds an **exposure-aware machine learning system** to classify
daily air-pollution **health risk levels (Low, Moderate, High)** using
multi-pollutant air quality data.

Instead of predicting AQI directly, the system focuses on **prolonged exposure,
temporal persistence, and volatility**, which are more relevant for real-world
health risk assessment.

---

## 🎯 Problem Statement
Air quality metrics like AQI compress complex pollutant information into a
single score and often fail to capture **sustained exposure effects**.

**Goal:**
Design an interpretable ML pipeline that:
- Models health risk based on exposure over time
- Avoids AQI leakage
- Prioritizes high-risk detection
- Remains explainable and robust

---

## 🗂 Dataset
- Source: Indian city-level air quality data
- Cities used: **20** (filtered for temporal reliability)
- Time span: ~1–5 years per city
- Core pollutants:
  - PM2.5, PM10, NO₂, SO₂, CO, O₃
- Industrial pollutants:
  - Benzene, Toluene, Xylene

Cities with fewer than one year of observations were excluded to ensure
reliable exposure modeling.

---

## 🧹 Data Cleaning & Preparation
- Removed physically impossible values
- Imputed missing core pollutants using **city-wise medians**
- Preserved missingness for sparsely measured industrial pollutants
- Maintained strict temporal ordering to prevent leakage

---

## 🧠 Feature Engineering

### Health Risk Score (HRS)
- Normalized pollutant concentrations (Min–Max scaling)
- Constructed a weighted composite score reflecting relative health impact
- Independent of AQI

### Exposure-Aware Features
- 3-day rolling exposure (short-term accumulation)
- Lag-1 dependency (temporal persistence)
- 7-day volatility (stability vs spikes)

### Context Features
- Month and day-of-week
- Winter indicator (India-specific domain knowledge)

---

## 🏷 Target Construction
- Labels derived from **7-day rolling exposure**
- Quantile-based thresholds:
  - Low (≤33%)
  - Moderate (33–66%)
  - High (>66%)
- Balanced class distribution (~33% each)

Labels represent **relative exposure-based risk**, not medical or regulatory
classifications.

---

## 🤖 Modeling & Evaluation
Models used:
- Logistic Regression (baseline)
- Random Forest (non-linear interactions)

Evaluation emphasized **class-wise recall**, with priority on detecting
**High-risk exposure periods**.

---

## 📊 Results
- High-risk recall: **~92%**
- Overall accuracy: **~89%**
- Near-zero misclassification of High risk as Low risk

Most errors occurred between adjacent classes (Moderate ↔ High),
indicating conservative and risk-aware model behavior.

---

## 🔍 Model Interpretation
Feature importance analysis showed:
1. Short-term accumulated exposure dominates predictions
2. Strong temporal persistence in pollution levels
3. Volatility provides secondary but meaningful signal
4. Seasonal context refines decisions

---

## ⚠ Limitations
- No direct health outcome data
- Pollutant weights are domain-informed, not learned
- Weather variables not included
- City-specific models not explored

---

## 🚀 Future Work
- Incorporate meteorological features
- Learn pollutant weights via optimization
- City-level or regional models
- Longer exposure windows (14–30 days)

---

## 📁 Project Structure
