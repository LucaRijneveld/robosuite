from stable_baselines3.common.env_checker import check_env
# from file_name import class_name
from CustomGym import RoboEnv # Import the environment you created, the names may be different

# Instantiate the environment
env = RoboEnv(RenderMode = False)
# Check the environment
check_env(env)
print('successsssss')