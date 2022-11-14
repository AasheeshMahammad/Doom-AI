from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class Tester():
    def __init__(self,model_path,env,render=True, config='./ViZDoom/scenarios/deadly_corridor.cfg'):
        self.model = PPO.load(model_path)
        self.env = env(render,config=config)
    
    def test(self,num_episodes=10):
        rewards,steps= evaluate_policy(self.model,self.env,n_eval_episodes=num_episodes,return_episode_rewards=True)
        mean_reward = sum(rewards)/len(rewards)
        for epi in range(len(rewards)):
            print(f"Reward for episode {epi+1} is {rewards[epi]}")
        print(f"Mean reward is {mean_reward}")
        

    def close(self):
        self.env.close()