from vizdoom import DoomGame
from gym import Env
from gym.spaces import Box,Discrete
from matplotlib import pyplot as plt
import cv2
import numpy as np
from stable_baselines3.common import env_checker
import cv2

class DoomEnvironment(Env): 
    
    def __init__(self, render=False, map_name="basic", config=None): 
        super().__init__()
        self.game = DoomGame()
        self.map_name=map_name
        if config == None:
            config =f"./ViZDoom/scenarios/{map_name}.cfg"
        self.game.load_config(config)
        self.game.set_window_visible(render)
        self.game.set_labels_buffer_enabled(True)
        self.game.init()
        state = self.game.get_state().labels_buffer
        original_shape = state.shape
        final_shape = self.reduce_size(state).shape
        self.observation_space = Box(low=0, high=255, shape=final_shape, dtype=np.uint8)
        self.buttons = self.game.get_available_buttons()
        #print(len(self.buttons))
        self.action_space = Discrete(len(self.buttons))
        self.reset_variables()
        
    def reset_variables(self):
        self.damage_taken = 0
        self.hitcount = 0
        self.damageCount = 0
        self.ammo = None
        self.itemCount = 0
        self.armor = 0
        self.health = 100
        self.position = [None, None]
        self.weapons = [None for _ in range(6)]
        self.prevWeapon = None
    
    def get_reward(self, reward,action):
        game_variables = self.game.get_state().game_variables
        #print(game_variables)
        size = len(game_variables)
        #print(self.map_name)
        if size == 1:
            self.ammo = game_variables[0]
            return reward
        elif self.map_name == "deadly_corridor":
            movement_reward = reward
            health, damage_taken, hitcount, ammo = game_variables
            if self.ammo == None:
                self.ammo = ammo
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            reward = movement_reward + damage_taken_delta*10 + hitcount_delta*200  + ammo_delta*5 
            return reward
        elif self.map_name == "deathmatch":
            killcount, health, armor, itemCount, damageCount ,selected_weapon, ammo,x,z = game_variables
            x,z = round(x,4), round(z,4)
            deltaInvalid = 0
            self.prevWeapon = selected_weapon
            if (x == self.position[0]) and (z == self.position[1]) and action >=2 and action <= 5:
                deltaInvalid -= 2
            self.position[0] = x; self.position[1] = z
            selected_weapon = int(selected_weapon)
            if self.weapons[selected_weapon-1] == None:
                self.weapons[selected_weapon-1] = ammo
            curAmmo = self.weapons[selected_weapon-1]
            deltaKill = killcount - self.hitcount
            deltaHealth = health - self.health
            deltaArmor = armor - self.armor
            deltaAmmo = ammo - curAmmo
            deltaDamageCount = damageCount - self.damageCount
            deltaItemCount = itemCount - self.itemCount
            self.itemCount = itemCount
            self.damageCount = damageCount
            self.armor = armor
            self.health = health
            self.weapons[selected_weapon-1] = ammo
            if deltaHealth > 0:
                deltaHealth *= 2
            if deltaDamageCount == 0 and deltaKill != 0:
                deltaKill = 0
            self.hitcount = killcount
            if deltaArmor > 0:
                deltaArmor *= 2
            reward = deltaArmor + deltaHealth + deltaKill*10 + deltaAmmo*2 + deltaDamageCount + deltaItemCount + reward + deltaInvalid
            #print(f"{deltaAmmo=}, {deltaHealth=}, {self.hitcount=},{deltaArmor=},{deltaDamageCount=}.{reward=},{deltaItemCount=}", end='\r')
            return reward
        elif self.map_name == "defend_the_line":
            ammo, health, killcount, damagecount = game_variables
            if self.ammo == None:
                self.ammo = ammo
            deltaHealth = health - self.health
            deltaAmmo = ammo - self.ammo
            deltaKill = killcount - self.hitcount
            deltaDamageCount = damagecount - self.damageCount
            if deltaDamageCount == 0 and deltaKill > 0:
                deltaKill = 0
            self.ammo = ammo
            self.health = health
            self.hitcount = killcount
            if deltaAmmo > 0:
                deltaAmmo = 0
            reward = deltaHealth*3 + reward + deltaKill*10 + deltaAmmo + deltaDamageCount
            return reward


        

    def step(self, action):
        actions = np.identity(len(self.buttons))
        reward = self.game.make_action(actions[action],2) 
        if self.game.get_state(): 
            state = self.game.get_state().labels_buffer
            state = self.reduce_size(state)
            #cv2.imshow("screen",state)
            #cv2.waitKey(10)
            reward = self.get_reward(reward,action)
            info = self.ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info 

    def render(): 
        pass
    
    def reset(self): 
        self.reset_variables()
        self.game.new_episode()
        state = self.game.get_state().labels_buffer
        state = self.reduce_size(state)
        return state


    def reduce_size(self, image):
        resize = cv2.resize(image,(120,120))
        x,y=resize.shape
        x_offset = 16
        y_offset = 0
        resize = resize[0:x-x_offset,0:y-y_offset]
        resize = np.reshape(resize,(x-x_offset,y-y_offset,1))
        return resize
    
    @staticmethod
    def show_difference(map_name):
        env = DoomEnvironment(map_name=map_name)
        state = env.game.get_state()
        reduced = env.reduce_size(state.labels_buffer)
        env.close()
        original = np.moveaxis(state.screen_buffer,0,-1)
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