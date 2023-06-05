import gym
from stable_baselines3 import PPO, DQN, A2C
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v0")
env = DummyVecEnv([lambda: env])

ppo_model = PPO("MlpPolicy", env, verbose=1)
dqn_model = DQN("MlpPolicy", env, verbose=1)
a2c_model = A2C("MlpPolicy", env, verbose=1)

ppo_model.learn(total_timesteps=10000)
dqn_model.learn(total_timesteps=10000)
a2c_model.learn(total_timesteps=10000)

ppo_mean_reward, _ = evaluate_policy(ppo_model, env, n_eval_episodes=10)
dqn_mean_reward, _ = evaluate_policy(dqn_model, env, n_eval_episodes=10)
a2c_mean_reward, _ = evaluate_policy(a2c_model, env, n_eval_episodes=10)

print(f"PPO Mean Reward: {ppo_mean_reward}")
print(f"DQN Mean Reward: {dqn_mean_reward}")
print(f"A2C Mean Reward: {a2c_mean_reward}")

obs = env.reset()

while True:
    action, _states = ppo_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
        break
obs = env.reset()
while True:
    action, _states = dqn_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
        break
obs = env.reset()
while True:
    action, _states = a2c_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
        break

env.close()
