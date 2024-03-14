# Add Webots controlling libraries
from controller import Robot
from controller import Supervisor


# Some general libraries
import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# Open CV
import cv2 as cv

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Stable_baselines3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


# Create an instance of robot
robot = Robot()

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("mps")



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.loss_values = []  # Add this line to store loss values


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True



class Environment(gym.Env, Supervisor):
    """The robot's environment in Webots."""
    
    def __init__(self):
        super().__init__()
                
        # General environment parameters
        self.max_speed = 1.5 # Maximum Angular speed in rad/s
        self.start_coordinate = np.array([-2.60, -2.96])
        self.destination_coordinate = np.array([-0.03, 2.72]) # Target (Goal) position
        self.reach_threshold = 0.09 # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1 # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([8, 8])
        
        
        # Activate Devices
        #~~ 1) Wheel Sensors
        self.left_motor = robot.getDevice('left wheel')
        self.right_motor = robot.getDevice('right wheel')

        # Set the motors to rotate for ever
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Zero out starting velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        #~~ 2) GPS Sensor
        sampling_period = 1 # in ms
        self.gps = robot.getDevice("gps")
        self.gps.enable(sampling_period)

        #~~ 3) Enable Camera
        sampling_period = 1 # in ms
        self.camera = robot.getDevice("camera")
        self.camera.enable(sampling_period)

        #~~ 4) Enable Touch Sensor
        self.touch = robot.getDevice("touch sensor")
        self.touch.enable(sampling_period)
              
        # # List of all available Distance sensors
        # available_devices = list(robot.devices.keys())
        # # Filter sensors name that contain 'so'
        # filtered_list = [item for item in available_devices if 'so' in item and any(char.isdigit() for char in item)]
        # filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

     
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        robot.step(200) # take some dummy steps in environment for initialization
        
        self.max_steps = 500
        
        #Distance Sensors
        # # Create dictionary from all available distance sensors and keep min and max of from total values
        # self.max_sensor = 0
        # self.min_sensor = 0
        # self.dist_sensors = {}
        # for i in filtered_list:    
        #     self.dist_sensors[i] = robot.getDevice(i)
        #     self.dist_sensors[i].enable(sampling_period)
        #     self.max_sensor = max(self.dist_sensors[i].max_value, self.max_sensor)    
        #     self.min_sensor = min(self.dist_sensors[i].min_value, self.min_sensor)
           
            
    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
        

    def get_distance_to_goal(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
    


    def get_camera_information(self):
        image = self.camera.getImageArray()
        image = np.array(image)
        # #     # Plot the image
        # plt.imshow(image)
        # plt.title('Image')
        # plt.xlabel('')
        # plt.ylabel('')
        # plt.show()
        # Convert image to a supported depth format (e.g., CV_8U)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
    
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        # # # Plot the image
        # plt.imshow(gray_image, cmap="gray")
        # plt.title('gray_image')
        # plt.xlabel('')
        # plt.ylabel('')
        # plt.show()
        # Calculate histogram
        histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    
        # # Display histogram
        # plt.plot(histogram)
        # plt.title('Histogram')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.show()
    
        # Calculate sum of frequencies before intensity 50 (obstacles)
        intensity_50 = 50
        target = histogram[:intensity_50].sum()
    
        # Calculate sum of additive values of histogram between intensities 240 to 255
        obstacle = histogram[200:].sum()
    
        return target, obstacle
    
        
    def get_distance_to_start(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_start = np.linalg.norm(self.start_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_start, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
    
        
    def get_current_position(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.

            
        position = self.gps.getValues()[0:2]
        position = np.array(position)

        normalized_current_position = self.normalizer(position, -4, +4)
        
        return normalized_current_position
    
    def get_observations(self):
        # """
        # Obtains and returns the normalized sensor data, current distance to the goal, and current position of the robot.
    
        # Returns:
        # - numpy.ndarray: State vector representing distance to goal, distance sensor values, and current position.
        # """
    
        # normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalized_current_distance = np.array([self.get_distance_to_goal()], dtype=np.float32)
        normalized_current_position = np.array(self.get_current_position(), dtype=np.float32)

        state_vector = np.concatenate([normalized_current_distance, normalized_current_position], axis=0)
        # state_vector = np.concatenate([normalized_current_distance, normalized_sensor_data], axis=0)

        return state_vector
    
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observations.
        
        Returns:
        - numpy.ndarray: Initial state vector.
        """
        # print("******")
        # print("self.simulationReset()")
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        # print("self.get_observations(): ", self.get_observations())
        return self.get_observations(), {}


    def step(self, action):    
        """
        Takes a step in the environment based on the given action.
        
        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """
        # print("action: ", action)
        self.apply_action(action)
        step_reward, done = self.get_reward()
        # print("step_reward: ", step_reward)
        state = self.get_observations() # New state
        # print("state: ", state)
        # Time-based termination condition
        if (int(self.getTime()) + 1) % self.max_steps == 0:
            done = True
        none = 0
        return state, step_reward, done, none, {}
        

    
    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.
        
        Returns:
        - The reward and done flag.
        """
        
        done = False
        reward = 0
        
        # normalized_sensor_data = self.get_sensor_data()
        normalized_current_distance = self.get_distance_to_goal()
        normalized_start_distance = self.get_distance_to_start()
        
        normalized_current_distance *= 100 # The value is between 0 and 1. Multiply by 100 will make the function work better
        reach_threshold = self.reach_threshold * 100

        distance_to_goal, distance_to_obstacles = self.get_camera_information()
        # print(distance_to_goal)
        # print(normalized_current_distance)

         # (1) Reward according to distance 
        if normalized_current_distance < 42:
            if normalized_current_distance < 5:
                growth_factor = 7
                A = 4.5            
            elif normalized_current_distance < 10:
                growth_factor = 7
                A = 3.5
            elif normalized_current_distance < 25:
                growth_factor = 6
                A = 2.5
            elif normalized_current_distance < 37:
                growth_factor = 4.5
                A = 2.2
            else:
                growth_factor = 3.2
                A = 1.9
            reward += A * (1 - np.exp(-growth_factor * (1 / normalized_current_distance)))
            
        else: 
            reward += -normalized_current_distance / 100
            
        # print(distance_to_obstacles)

        # (3) Punish if close to obstacles
        if distance_to_obstacles > 1700:
                
                if distance_to_obstacles > 4000:
                    reward-= 3
                    # print("rew-40")
                elif distance_to_obstacles > 3000:
                    reward-= 2
                    # print("rew-25")
                elif distance_to_obstacles > 2500:
                    reward-= 2
                    # print("rew-15")
                elif distance_to_obstacles > 2000:
                    reward-= 1
                    # print("rew-10")
        elif distance_to_obstacles < 1000:
                if distance_to_obstacles < 800:
                    reward+= 3
                    # print("rew+3")
                elif distance_to_obstacles < 1000:
                    reward+= 2
                    # print("rew+2")

        
        # (3) Punish if close to Goal
        if distance_to_goal > 0:
                
            if distance_to_goal > 1500:
                reward+= 5
            elif distance_to_goal > 1000:
                reward+= 4
            elif distance_to_goal > 700:
                reward+= 3
            elif distance_to_goal > 600:
                reward+= 3
            elif distance_to_goal > 500:
                reward+= 2
            elif distance_to_goal > 400:
                reward+= 2
            elif distance_to_goal > 300:
                reward+= 2
            elif distance_to_goal > 200:
                reward+= 2
            elif distance_to_goal > 100:
                reward+= 1

        # (2) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value

        if normalized_current_distance < reach_threshold:
            # Reward for finishing the task
            done = True
            reward += 500
            print('+++ SOlVED +++')
        elif check_collision:
            # Punish if Collision
            done = True
            reward -= 7

        return reward, done


    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        if action == 0: # move forward
            # print("forward")
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 1: # turn right
            # print("right")
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(-self.max_speed)
        elif action == 2: # turn left
            # print("left")
            self.left_motor.setVelocity(-self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        
        robot.step(500)

        
        self.left_motor.setPosition(0)
        self.right_motor.setPosition(0)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)           
    

class Agent_FUNCTION():
    def __init__(self, save_path, load_path, num_episodes, max_steps, 
                  learning_rate, gamma, hidden_size, clip_grad_norm, baseline):
        self.save_path = save_path
        self.load_path = load_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.clip_grad_norm = clip_grad_norm
        self.baseline = baseline


        self.env = Environment()
        self.env = Monitor(self.env, "tmp/")

        #PPO
        self.policy_network = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log="./results_tensorboard/")#.learn(total_timesteps=1000)
        
    
    # def save(self, path):
    #     print(self.save_path ,"+PPO2-Best")
    #     self.policy_network.save(self.save_path + "+PPO2-Best")
    #     # torch.save(self.policy_network.state_dict(), self.save_path + path)

    def load(self):
        print(self.save_path+"best_model")
        # self.policy_network = PPO.load(self.save_path+"deepq_cartpole")
        self.policy_network = PPO.load("/Users/narges/Desktop/Term 3/AMR/HW/HW2/HW-2-2_V10/controllers/my_controller/tmp/best_model.zip")

        # self.policy_network.load_state_dict(torch.load(self.load_path, map_location=torch.device('cpu')))

    def compute_returns(self, rewards):
        t_steps = torch.arange(len(rewards))
        discount_factors = torch.pow(self.gamma, t_steps).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        returns = rewards * discount_factors
        returns = returns.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))
        if self.baseline:
            mean_reward = torch.mean(rewards)
            returns -= mean_reward
        return returns

    def compute_loss(self, log_probs, returns):
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        return torch.stack(loss).sum()


    def train(self, total_timesteps) :

        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # self.policy_network.train(gradient_steps=5)
        start_time = time.time()
        reward_history = []
        best_score = -np.inf

        self.policy_network.learn(total_timesteps=total_timesteps, callback=callback)
        self.env.reset()


 
    def test(self):
        
        state = self.env.reset()
        for episode in range(1, self.num_episodes+1):
            rewards = []
            # state = self.env.reset()
            done = False
            ep_reward = 0
            state=np.array(state[0])
            while not done:
                # Ensure the state is in the correct format (convert to numpy array if needed)
                # print("state: ", state)
                state_np = np.array(state)

                # Get the action from the policy network
                action, _ = self.policy_network.predict(state_np)

                # Take the action in the environment
                state, reward, done, _,_ = self.env.step(action)
                # print("reward: ", reward)
                ep_reward += reward

            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
            state = self.env.reset()
        
            
            
if __name__ == '__main__':
    # Parameters
    save_path = './results/'   
    load_path = './results/final_weights.pt'
    train_mode = False 
    num_episodes = 2000 if train_mode else 10
    max_steps = 100 if train_mode else 500
    learning_rate = 0.01
    gamma = 0.99
    hidden_size = 7
    clip_grad_norm = 5
    baseline = True

    total_timesteps = 400000


    env = Environment()

    agent =Agent_FUNCTION(save_path, load_path, num_episodes, max_steps, 
                            learning_rate, gamma, hidden_size, clip_grad_norm, baseline)


    if train_mode:
        # Initialize Training
        agent.train(total_timesteps=total_timesteps)
    else:
        # Test
        agent.load()
        agent.test()
