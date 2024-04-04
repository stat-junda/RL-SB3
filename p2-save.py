import gym
import os
from stable_baselines3 import PPO, A2C

models_dir = "models/PPO"   # if in the terminal, just use "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
    

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2

env.reset()

model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=logdir) 

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO") #reset_num_timesteps=False表示不重置训练过程中的总时间步数计数器
    model.save(f"{models_dir}/{TIMESTEPS*i}")


    