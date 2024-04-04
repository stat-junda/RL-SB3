import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

models_dir = "models/PPO"
# model_dir = f"{models_dir}/PPO-{int(time.time())}"
#logdir = f"logs/PPO-{int(time.time())}"

env = make_vec_env('LunarLander-v2', n_envs=1)  # continuous: LunarLanderContinuous-v2
# env = gym.make('LunarLander-v2')
# env = DummyVecEnv([lambda: env])  # 将环境包装成VecEnv

env.reset()

model_path = f"{models_dir}/260000.zip"

model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Add deterministic=True for consistent behavior
        obs, reward, done, info = env.step(action)
        env.render(mode='human')  # Ensure env.render() is called with mode='human' if necessary
        print(f"Episode: {ep+1}, Reward: {reward}")
        time.sleep(0.01)
       
env.close()

    