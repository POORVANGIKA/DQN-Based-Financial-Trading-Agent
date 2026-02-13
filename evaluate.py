import numpy as np
import matplotlib.pyplot as plt

def evaluate_models():
    models = ['DQN RL', 'LSTM', 'ARIMA']
    returns = [12.5, 7.8, 3.5]  # Example cumulative returns (%)
    sharpe = [1.5, 1.1, 0.8]
    max_drawdown = [8.0, 12.3, 15.7]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(x - width, returns, width, label='Return (%)')
    plt.bar(x, sharpe, width, label='Sharpe Ratio')
    plt.bar(x + width, max_drawdown, width, label='Max Drawdown (%)')

    plt.xticks(x, models)
    plt.ylabel('Metric Value')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_models()
