import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from snakeenvp4 import SnekEnv

models_dir = "models/1711245323"
# model_dir = f"{models_dir}/PPO-{int(time.time())}"
#logdir = f"logs/PPO-{int(time.time())}"

env = SnekEnv()  # continuous: LunarLanderContinuous-v2
# env = gym.make('LunarLander-v2')
# env = DummyVecEnv([lambda: env])  # 将环境包装成VecEnv

env.reset()

model_path = f"{models_dir}/120000.zip"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Add deterministic=True for consistent behavior
        obs, reward, done, info = env.step(action)
        print(f"Episode: {ep+1}, Reward: {reward}")
       
env.close()

    