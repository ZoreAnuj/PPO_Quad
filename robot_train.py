from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pybullet_envs.robot_env import RobotEnv


# Create the vectorized environment
env = make_vec_env(lambda: RobotEnv("urdf/quad.urdf", show_training = True), n_envs=1)

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1, learning_rate=5e-5, n_steps=4096, batch_size=256)

# Train the model
model.learn(total_timesteps = 100_000)

# Save the model
model.save("ppo_robot")

# Load the model 
model = PPO.load("ppo_robot")

env.close()
