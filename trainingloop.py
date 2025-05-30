import gym
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO  
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import RecordVideo

#training model
'''params include:
    - model: the model (specified in main function)
    - timesteps: total number of env steps
    - save_every: controls how frequirement the model is trained and evaluated
    - model_dir: directory where model file is going to be (specified in main function)
    - log_dir: directory where log file is going to be (specified in main function)
    - env: environment (specified in main function)'''

def train(model,timesteps=0,save=0,model_dir=None,log_dir=None,env=None):
    start = time.time()
    stats = []

    for step in range(0,timesteps,save):
        model.learn(total_timesteps=save, reset_num_timesteps=False)
        model.save(f"{model_dir}/ppo_{step}")

        mean, standard = evaluate_policy(model,env, n_eval_episodes=1)
        elapsed = time.time() - start
        stats.append((step,save,mean,standard,elapsed))
    
    with open(f'{log_dir}/trainingStats.csv','w',newline='') as f:
        write = csv.writer(f)
        write.writerow(['step','mean reward', 'standard reward', 'elapsed time'])
        write.writerow(stats)
    
    return stats

#recording steps
'''params include:
    - env_id: id for the environment
    - model: the model (specified in main function)
    - length: total number os steps function will record
    - video_dir directory video will be saved (specified in main function)
'''
def record(env_id,model,length,video_dir):
    env = gym.make(env_id)
    env = RecordVideo(env,video_dir,episode_trigger=lambda e: True)
    o = env.reset() 

    for _ in range(length):
        action,_ = model.predict(o)
        o,_,finished,_ = env.step(action)
        if (finished):
            o = env.reset()
    env.close()

#function for evaluation
'''params include:
    - model: the model (specified in main function)
    - env_name: name of environment (specified in main function)
    - episodes: number of episodes 
'''

def evaluate(model,env_name,episodes=0):
    env = gym.make(env_name)
    results = []

    for e in range(episodes):
        o = env.reset()
        finished = False
        score = 0
        moves = 0
        actions = []
        start = time.time()

        while not finished:
            action,_ = model.predict(o)
            o,reward,finished,_ = env.step(action)
            score += reward
            moves+=1
            actions.append(action)
        
        elapsed = time.time()- start
        results.append((e,score,moves,elapsed,actions))
    
    with open(f'{log_dir}/evalStats.csv','w',newline="") as f:
        write = csv.writer(f)
        write.writerow(['episode','score','num moves','elapsed time','actions'])
        for row in results:
            write.writerow([row[0], row[1], row[2], row[3], row[4]])
    
    return results

if __name__ == "main":
    env_name = None #change name to env name when done
    log_dir = '/logs'
    model_dir = "/models"
    video_dir = "/videos"

    os.makedirs(log_dir)
    os.makedirs(model_dir)
    os.makedirs(video_dir)

    #When environment is created, put it here 
    env = None
    #put correct policy name, as is currently set to none
    model = PPO('None',env,verbose=1)

