

A Deep Reinforcement Learning project that implements a \*\*Deep Q-Network (DQN)\*\* for automated stock trading strategy optimization.



This project compares Reinforcement Learning approaches with traditional time-series models such as \*\*ARIMA\*\* and \*\*LSTM\*\*.



---



\## 🚀 Project Overview



Financial markets are dynamic and complex. Traditional models like ARIMA and LSTM capture patterns in historical price data, but they do not actively learn trading strategies.



This project implements:



\- 📌 Deep Q-Network (DQN) for decision-based trading

\- 📌 LSTM for sequence modeling

\- 📌 ARIMA for statistical forecasting

\- 📌 Data preprocessing pipeline

\- 📌 Evaluation metrics for performance comparison



The goal is to evaluate how reinforcement learning performs compared to traditional forecasting models.



---



\## 🧠 Technologies Used



\- Python 3.10

\- NumPy

\- Pandas

\- Matplotlib

\- Gym / Gymnasium

\- PyTorch / TensorFlow

\- Scikit-learn



---



\## 📂 Project Structure



```

DQN-Based-Financial-Trading-Agent/

│

├── preprocessing.py      # Data cleaning and feature engineering

├── train\_dqn.py          # DQN training script

├── train\_lstm.py         # LSTM model training

├── train\_arima.py        # ARIMA baseline model

├── evaluate.py           # Performance evaluation

├── requirements.txt

└── README.md

```



---



\## ⚙️ Installation \& Setup



Clone the repository:



```bash

git clone https://github.com/POORVANGIKA/DQN-Based-Financial-Trading-Agent.git

cd DQN-Based-Financial-Trading-Agent

```



Create virtual environment:



```bash

python -m venv venv

venv\\Scripts\\activate   # Windows

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run training:



```bash

python train\_dqn.py

```



---



\## 📊 Models Implemented



\### 1️⃣ Deep Q-Network (DQN)

\- Learns optimal buy/sell/hold policy

\- Uses reward-based learning

\- Suitable for sequential decision problems



\### 2️⃣ LSTM

\- Captures temporal dependencies

\- Useful for price trend forecasting



\### 3️⃣ ARIMA

\- Classical statistical time-series forecasting model



---



\## 📌 Future Improvements



\- Add transaction cost modeling

\- Implement Double DQN

\- Add PPO / A2C comparison

\- Add visualization dashboard

\- Deploy as web application



---



\## 👩‍💻 Author



\*\*POORVANGIKA\*\*



---



⭐ If you found this project useful, feel free to star the repository.

