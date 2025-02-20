# Yield Analysis with Hidden Markov Model

## Overview
This project implements a **Hidden Markov Model (HMM)** to analyze **yield curves** and classify market states based on multiple economic, policy, and statistical indicators. The model integrates:

- **Federal Reserve policy periods**
- **Yield curve shapes**
- **Business cycles**
- **Crisis indicators**

By combining these factors, the model generates **comprehensive market state classifications** to assess financial conditions over time.

## Features
- Loads and preprocesses **yield curve data**
- Calculates **returns and volatility metrics**
- Implements multiple **classification schemes**:
  - **Federal Reserve Policy Periods** (QE/QT)
  - **Yield Curve Shapes** (Normal/Flat/Inverted)
  - **Business Cycles** (Expansion/Peak/Recession/Recovery)
  - **Crisis Periods**
- Computes **rolling z-scores** for volatility
- Generates **final market state scores** using weighted combinations
- Calculates **state transition probabilities**
- Visualizes results with **annotated plots**

---

## Classification Schemes
### **Federal Reserve Policy Periods**
- **Pre-COVID QE**: *2019-12-26 to 2020-03-15*
- **COVID QE**: *2020-03-15 to 2022-03-01*
- **QT1**: *2017-10-01 to 2019-07-31*
- **QT2**: *2022-06-01 to 2024-12-31*

### **Yield Curve Shapes**
- **Pre-COVID Normal**: *2019-12-26 to 2020-02-18*
- **COVID Inversion**: *2020-02-19 to 2020-04-30*
- **Post-COVID Steepening**: *2020-05-01 to 2021-12-31*
- **Flattening Trend**: *2022-01-01 to 2022-12-31*
- **QT Flat or Inverted**: *2023-01-01 to 2024-12-26*

### **Business Cycles**
- **Expansion**: *2019-12-26 to 2020-02-18*
- **Recession**: *2020-02-19 to 2020-04-30*
- **Recovery**: *2020-05-01 to 2024-12-26*

### **Crisis Periods**
- **COVID Crisis**: *2020-02-19 to 2020-04-30*

---

## State Classification Method
The final **market state classification** is determined using a weighted combination of three scores:

### **1. Statistical Score (60% weight)**
- **Low**: *z < 0.5* (score: `0.3`)
- **Medium**: *0.5 ≤ z ≤ 1.5* (score: `0.6`)
- **High**: *z > 1.5* (score: `0.9`)

### **2. Market Score (25% weight)**
- **QE periods** score **lower** (`0.2`)
- **QT periods** score **higher** (`0.7`)
- Normalized with **yield curve shape score**

### **3. Economic Score (15% weight)**
- Combines **business cycle** and **crisis indicators**
- Scored based on **cycle phase** and **crisis presence**

### **Final State Classification**
- **Low**: *score < 0.4*
- **Medium**: *0.4 ≤ score ≤ 0.7*
- **High**: *score > 0.7*

---

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install pandas numpy matplotlib statsmodels decimal
```

---

## Usage
### **1. Load and Process Data**
```python
# Load data
df = load_data('your_data.csv')

# Calculate financial metrics
df = calculate_returns(df)
df = calculate_volatility(df)
df = classify_dates(df)
df = calculate_rolling_z_scores(df)
df = calculate_final_score(df)
```

### **2. Generate Visualization**
```python
plot_yield_states(df)
```

---

## Output
The code produces a **visual representation** of the yield curve analysis, including:
- **Yield rates over time**
- **State classifications** (color-coded points)
- **Transition probabilities** between states
- **State labels and legend**

---

## Notes
- The model applies a **21-day rolling window** for **volatility calculations**.
- **Markov chain principles** are used to compute **state transition probabilities**.
- **State overrides** are applied in cases of **extreme market conditions**.

## Author
**Cem Koymen**  
