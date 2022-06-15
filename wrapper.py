from vizdoom import DoomGame
from gym import Env
from gym.spaces import Box,Discrete
from matplotlib import pyplot as plt
import cv2
import numpy as np
from stable_baselines3.common import env_checker

class DoomEnvironment(Env): 
    
    def __init__(self, render=False): 
        super().__init__()
        self.game = DoomGame()
        self.game.load_config('./VizDoom/scenarios/basic.cfg')
        self.game.set_window_visible(render)
        self.game.init()
        state = self.game.get_state().screen_buffer
        original_shape = state.shape
        final_shape = self.reduce_size(state).shape
        self.observation_space = Box(low=0, high=255, shape=final_shape, dtype=np.uint8) 
        self.action_space = Discrete(3)


    def step(self, action):
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4) 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.reduce_size(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info 

    def render(): 
        pass
    
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = self.reduce_size(state)
        return state


    def reduce_size(self, image):
        shiftedGreyImage = cv2.cvtColor(np.moveaxis(image,0,-1),cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(shiftedGreyImage,(120,120),interpolation=cv2.INTER_CUBIC)
        x,y=resize.shape
        x_offset = 16
        y_offset = 0
        resize = resize[0:x-x_offset,0:y-y_offset]
        resize = np.reshape(resize,(x-x_offset,y-y_offset,1))
        return resize
    
    @staticmethod
    def show_difference():
        env = DoomEnvironment()
        state = env.game.get_state().screen_buffer
        reduced = env.reduce_size(state)
        env.close()
        original = np.moveaxis(state,0,-1)
        fig = plt.figure(figsize=(10, 8))
        fig.add_subplot(2,2,1)
        plt.imshow(original)
        axis = "on"
        plt.axis(axis)
        plt.title('Original Image')
        fig.add_subplot(2,2,2)
        plt.imshow(reduced)
        plt.axis(axis)
        plt.title('Processed Image')
    
    @staticmethod
    def check_environment():
        env = DoomEnvironment()
        try:
            env_checker.check_env(env)
            print("Environmet is Good to go")
        except Exception as e:
            print(e)
        finally:
            env.close()

    def close(self): 
        self.game.close()