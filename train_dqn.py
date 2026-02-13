import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import TradingEnv
import os

def train_dqn(processed_data_file='data/nifty100_processed.feather'):
    if not os.path.exists(processed_data_file):
        print(f"Processed data file {processed_data_file} not found! Run preprocessing.py first.")
        return

    print("Loading processed data...")
    data = pd.read_feather(processed_data_file).values

    train_size = int(0.8 * len(data))
    train_data = data[:train_size]

    env = DummyVecEnv([lambda: TradingEnv(train_data)])

    print("Initializing DQN model...")
    model = DQN('MlpPolicy', env, verbose=1)

    print("Training model for 100,000 timesteps...")
    model.learn(total_timesteps=100000)

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'dqn_nifty100_10years.zip')
    model.save(model_path)

    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    train_dqn()
