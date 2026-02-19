# 📈 DQN-Based Financial Trading Agent

A Deep Reinforcement Learning project that implements a **Deep Q-Network (DQN)** to optimize automated stock trading strategies.
The project also compares Reinforcement Learning with traditional time-series models such as **ARIMA** and **LSTM** for performance evaluation.

---

## 🚀 Project Overview

Financial markets are highly dynamic and uncertain. Traditional forecasting models can capture historical patterns but do not actively learn trading decisions.

This project builds an intelligent trading agent capable of learning **buy, sell, and hold strategies** directly from market data through interaction with a custom trading environment.

### 🔍 Key Features

* Deep Q-Network (DQN) for decision-based trading
* Custom Gym trading environment
* LSTM model for sequential price prediction
* ARIMA statistical forecasting baseline
* Technical indicator engineering (RSI, SMA)
* Performance evaluation using financial metrics

---

## 🧠 Technologies Used

* Python 3.10
* NumPy & Pandas
* Matplotlib
* Gym / Gymnasium
* Stable-Baselines3
* TensorFlow / Keras
* Statsmodels
* Scikit-learn

---

## 📂 Project Structure

```
DQN-Based-Financial-Trading-Agent/
│
├── preprocessing.py      # Data cleaning & feature engineering
├── env.py                # Custom trading environment
├── train_dqn.py          # DQN training pipeline
├── train_lstm.py         # LSTM forecasting model
├── train_arima.py        # ARIMA baseline model
├── evaluate.py           # Model performance comparison
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/POORVANGIKA/DQN-Based-Financial-Trading-Agent.git
cd DQN-Based-Financial-Trading-Agent
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Preprocess Data

```bash
python preprocessing.py
```

### Train DQN Agent

```bash
python train_dqn.py
```

### Train LSTM Model

```bash
python train_lstm.py
```

### Run ARIMA Baseline

```bash
python train_arima.py
```

### Evaluate Performance

```bash
python evaluate.py
```

---

## 📊 Models Implemented

### 🔹 Deep Q-Network (DQN)

* Learns optimal trading policy (Buy / Sell / Hold)
* Reward-based sequential decision learning
* Suitable for adaptive trading environments

### 🔹 LSTM

* Captures temporal dependencies in price movements
* Effective for trend prediction

### 🔹 ARIMA

* Classical statistical time-series forecasting model
* Serves as a baseline for comparison

---

## 📌 Future Improvements

* Transaction cost and slippage modeling
* Double DQN implementation
* PPO / A2C reinforcement learning comparison
* Interactive visualization dashboard
* Web deployment of trading agent

---

## 👩‍💻 Author

**Poorvangika Kanwar**

---

⭐ If you found this project useful, consider giving it a star!


