# 🧪 Machine Learning Hackathon — Chemical Engineering Yield Prediction

A machine learning project developed for the **Machine Learning Hackathon of Fugacity 2026**, organized by the **Indian Institute of Technology (IIT), Kharagpur**.

The objective of this project is to predict the **overall yield of a chemical engineering process** using process parameters such as flow rate, concentration, temperature, and reactor length.

---

## 📌 Problem Statement

Modern chemical plants generate large amounts of data from sensors measuring different process parameters.

In this hackathon, the task was to build a machine learning model capable of predicting **overall chemical process yield** from a curated dataset containing process and operating parameters.

The project focuses on applying:

- Data preprocessing
- Feature engineering
- Regression modeling
- Ensemble learning
- Prediction generation

The final model generates predictions that can be submitted to the hackathon leaderboard.

---

## 🎯 Objective

The primary objective is to predict:

> **`overall_yield`**

using the available chemical process parameters.

### Input Features

The dataset contains the following process variables:

| Feature | Description |
|---|---|
| `flow_rate_L_min` | Flow rate in liters per minute |
| `concentration_mol_L` | Concentration in mol/L |
| `inlet_temperature_K` | Inlet temperature in Kelvin |
| `length_m` | Process/reactor length in meters |
| `jacket_temperature_K` | Jacket temperature in Kelvin |

### Target

```text
overall_yield
