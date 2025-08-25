"""
Random Agent for Mario Bros Game using Stable-Baselines3 Wrappers

This script creates a Mario Bros environment using the stable-baselines3 MaxAndSkipEnv
wrapper and runs a random agent to demonstrate the game mechanics.
"""

import gymnasium as gym
import numpy as np
import ale_py
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class TimeLimitWrapper(gym.Wrapper):
    """
    Wrapper that limits the maximum number of steps per episode to prevent infinite games.
    
    Args:
        env: The gymnasium environment to wrap
        max_steps: Maximum steps allowed per episode (default: 10000)
    """
    def __init__(self, env, max_steps=10000):
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        """Reset the environment and step counter"""
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Take a step and check if time limit is reached"""
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Force episode to end if we've taken too many steps
        if self.current_step >= self.max_steps:
            truncated = True
            info['time_limit_reached'] = True
            info['current_step'] = self.current_step  # Consistent naming
        
        return obs, reward, terminated, truncated, info

def create_mario_env(render_mode='human', max_steps=10000, frame_skip=4):
    """
    Create a Mario Bros environment with standard wrappers.
    
    Args:
        render_mode: 'human' for visual display, None for no rendering
        max_steps: Maximum steps per episode  
        frame_skip: Number of frames to skip per action
        
    Returns:
        Wrapped gymnasium environment
    """
    # Create base environment
    env = gym.make('ALE/MarioBros-v5', render_mode=render_mode)
    
    # Apply wrappers
    env = TimeLimitWrapper(env, max_steps=max_steps)              # Custom time limit
    env = MaxAndSkipEnv(env, skip=frame_skip)                     # Stable-baselines3 frame skipping
    
    return env

def run_random_agent(episodes=None, max_steps_per_episode=10000):
    """
    Run a random agent on Mario Bros using stable-baselines3 wrappers.
    
    Args:
        episodes: Number of episodes to run (None = infinite)
        max_steps_per_episode: Maximum steps per episode
    """
    print("🎮 Mario Bros Random Agent (with Stable-Baselines3 wrappers)")
    print("=" * 60)
    
    # Create environment
    print("Loading Mario Bros environment...")
    env = create_mario_env(max_steps=max_steps_per_episode)
    
    # Get initial observation
    obs, info = env.reset()
    print(f"✅ Environment loaded successfully!")
    print(f"🖼️  Observation shape: {obs.shape}")
    print(f"🎯 Action space: {env.action_space} ({env.action_space.n} possible actions)")
    print(f"⏱️  Max steps per episode: {max_steps_per_episode}")
    print(f"🎬 Using Stable-Baselines3 MaxAndSkipEnv (4 frame skip)")
    print("\n🎮 Starting random gameplay... (Press Ctrl+C to stop)")
    print("-" * 60)
    
    # Game statistics
    episode_count = 0
    total_steps = 0
    step_count = 0
    episode_rewards = []
    current_reward = 0
    
    try:
        while episodes is None or episode_count < episodes:
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            total_steps += 1
            current_reward += reward
            
            # Check if episode ended
            if terminated or truncated:
                episode_count += 1
                episode_rewards.append(current_reward)
                
                # Determine end reason
                if info.get('time_limit_reached', False):
                    end_reason = "time limit"
                elif terminated:
                    end_reason = "game over (Mario died)"
                else:
                    end_reason = "episode complete"
                
                # Print episode summary
                print(f"📊 Episode {episode_count}: {step_count} steps, "
                      f"reward: {current_reward:.1f}, ended by: {end_reason}")
                
                # Reset for next episode
                obs, info = env.reset()
                step_count = 0
                current_reward = 0
            
            # Print progress during long episodes
            elif step_count > 0 and step_count % 1000 == 0:
                print(f"   🏃 Step {step_count} in episode {episode_count + 1}, "
                      f"current reward: {current_reward:.1f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    finally:
        # Print final statistics
        print("\n" + "=" * 60)
        print("📈 Final Statistics:")
        print(f"   Episodes completed: {episode_count}")
        print(f"   Total steps: {total_steps}")
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            max_reward = np.max(episode_rewards)
            min_reward = np.min(episode_rewards)
            print(f"   Average reward per episode: {avg_reward:.1f}")
            print(f"   Best episode reward: {max_reward:.1f}")
            print(f"   Worst episode reward: {min_reward:.1f}")
        print("🎮 Thanks for playing!")
        
        env.close()

def test_environment():
    """Quick test to verify the environment works correctly"""
    print("🧪 Testing Mario Bros environment...")
    
    try:
        env = create_mario_env(render_mode=None)  # No rendering for test
        obs, info = env.reset()
        
        print(f"✅ Environment created successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        print(f"   Action space: {env.action_space}")
        
        # Take a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✅ Environment test completed successfully!")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False
    
    return True

def main():
    """Main function - run random agent indefinitely"""
    # Optional: run environment test first
    if not test_environment():
        return
    
    print("\n" + "="*60)
    
    # Run the random agent
    run_random_agent(episodes=None)  # Run forever until Ctrl+C

if __name__ == "__main__":
    main()