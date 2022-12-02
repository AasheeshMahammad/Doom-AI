from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time


class Tester():
    def __init__(self,model_path,env,render=True, map_name="basic",config='./ViZDoom/scenarios/deadly_corridor.cfg'):
        self.model = PPO.load(model_path)
        self.env = env(render,map_name)
    
    def do(self, num_episodes):
        for episode in range(num_episodes): 
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done: 
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                #time.sleep(0.10)
                total_reward += reward
            print('Total Reward for episode {} is {}'.format(episode+1, total_reward))
            time.sleep(1)

    def test(self,num_episodes=5):
        #rewards,steps= evaluate_policy(self.model,self.env,n_eval_episodes=num_episodes,return_episode_rewards=True)
        self.do(num_episodes)
        '''mean_reward = sum(rewards)/len(rewards)
        for epi in range(len(rewards)):
            print(f"Reward for episode {epi+1} is {rewards[epi]}")
        print(f"Mean reward is {mean_reward}")'''
        

    def close(self):
        self.env.close()