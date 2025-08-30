import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def watch_agent(agent_type='trained', model_path=None):
    """
    Loads and runs an agent in a visible environment.

    Args:
        agent_type (str): The type of agent to run ('trained' or 'random').
        model_path (str): The path to the saved .zip model file (required for 'trained' agent).
    """
    model = None
    # --- 1. Load the model if we are using a trained agent ---
    if agent_type == 'trained':
        if not model_path or not os.path.exists(model_path):
            print(f"Error: Model not found at '{model_path}'")
            print("Please specify a valid path for the trained model.")
            return
        
        print(f"Loading model from: {model_path}")
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"Error loading the model: {e}")
            return
    elif agent_type != 'random':
        print(f"Error: Invalid agent_type '{agent_type}'. Choose 'trained' or 'random'.")
        return

    # --- 2. Create the environment for visualization ---
    env_id = 'ALE/MarioBros-v5'
    env = gym.make(env_id, render_mode='human')
    
    # --- 3. Apply the SAME wrappers as in training ---
    env = AtariWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    print(f"✅ Environment loaded successfully.")
    print(f"🚀 Starting {agent_type.upper()} agent... Press Ctrl+C in the terminal to stop.")
    
    # --- 4. Run the game loop ---
    obs = env.reset()
    
    try:
        while True:
            # --- 5. Get the action from the chosen agent type ---
            if agent_type == 'trained':
                # Get action from the trained model
                action, _states = model.predict(obs, deterministic=True)
            else: # 'random' agent
                # Get a random action from the environment's action space
                action = [env.action_space.sample()]

            # Take the action in the environment
            obs, reward, done, info = env.step(action)
            
            if done[0]:
                print("Episode finished. Resetting...")
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    
    finally:
        # --- 6. Clean up ---
        env.close()
        print("✅ Environment closed. Goodbye!")


if __name__ == '__main__':
    # --- CHOOSE AGENT MODE ---
    # Set to 'TRAINED' to watch your PPO agent.
    # Set to 'RANDOM' to watch a random agent for comparison.
    AGENT_MODE = 'TRAINED'
    
    # --- DEFINE THE PATH TO YOUR MODEL ---
    # This is only used if AGENT_MODE is 'TRAINED'.
    MODEL_PATH = "mario_training_logs/best_model.zip"
    
    if AGENT_MODE.upper() == 'TRAINED':
        watch_agent(agent_type='trained', model_path=MODEL_PATH)
    elif AGENT_MODE.upper() == 'RANDOM':
        watch_agent(agent_type='random')
    else:
        print(f"Invalid AGENT_MODE: '{AGENT_MODE}'. Please choose 'TRAINED' or 'RANDOM'.")

