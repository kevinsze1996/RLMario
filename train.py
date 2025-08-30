import gymnasium as gym
import numpy as np
import os
import shutil 
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import LoadMonitorResultsError
import ale_py

class SaveAndSyncCallback(BaseCallback):
    """
    A callback that saves the best model and periodically syncs
    local log folders to a permanent location.
    """
    def __init__(self, check_freq: int, log_dir: str, board_dir: str, save_dir: str, sync_freq: int = 10000, verbose: int = 1):
        super(SaveAndSyncCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.sync_freq = sync_freq
        self.log_dir = log_dir
        self.board_dir = board_dir
        self.save_dir = save_dir
        
        self.best_model_save_path = os.path.join(self.save_dir, 'best_model.zip')
        self.permanent_monitor_path = os.path.join(self.save_dir, 'monitor_logs')
        self.permanent_board_path = os.path.join(self.save_dir, 'board_logs')
        
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.permanent_monitor_path, exist_ok=True)
        os.makedirs(self.permanent_board_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.best_model_save_path}")
                        self.model.save(self.best_model_save_path)
            except (LoadMonitorResultsError, FileNotFoundError):
                if self.verbose > 0:
                    print(f"Could not find log file at {self.log_dir}, skipping check.")
        
        if self.n_calls % self.sync_freq == 0:
            if self.verbose > 0:
                print(f"Syncing local logs to {self.save_dir}")
            
            if os.path.exists(self.permanent_monitor_path):
                shutil.rmtree(self.permanent_monitor_path)
            if os.path.exists(self.log_dir):
                shutil.copytree(self.log_dir, self.permanent_monitor_path)

            if os.path.exists(self.permanent_board_path):
                shutil.rmtree(self.permanent_board_path)
            if os.path.exists(self.board_dir):
                shutil.copytree(self.board_dir, self.permanent_board_path)

        return True

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    local_monitor_dir = "tmp/"
    local_board_dir = "board/"
    permanent_log_dir = "mario_training_logs/"
    
    permanent_monitor_path = os.path.join(permanent_log_dir, 'monitor_logs')
    permanent_board_path = os.path.join(permanent_log_dir, 'board_logs')
    if os.path.exists(permanent_monitor_path):
        print("Found old monitor logs, restoring to local session...")
        shutil.copytree(permanent_monitor_path, local_monitor_dir, dirs_exist_ok=True)
    if os.path.exists(permanent_board_path):
        print("Found old TensorBoard logs, restoring to local session...")
        shutil.copytree(permanent_board_path, local_board_dir, dirs_exist_ok=True)

    env_id = 'ALE/MarioBros-v5'
    num_cpu = 4
    
    env = make_atari_env(env_id, n_envs=num_cpu, seed=0, monitor_dir=local_monitor_dir, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    
    best_model_path = os.path.join(permanent_log_dir, "best_model.zip")
    TARGET_TIMESTEPS = 10000000
    
    if os.path.exists(best_model_path):
        print("------------- Found best model. Resuming training. -------------")
        model = PPO.load(best_model_path, env=env)
    else:
        print("------------- No model found. Starting new training. -------------")
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=local_board_dir, learning_rate=0.00003)
    
    print("------------- Start Learning -------------")
    callback = SaveAndSyncCallback(check_freq=1000, log_dir=local_monitor_dir, board_dir=local_board_dir, save_dir=permanent_log_dir)
    
    remaining_timesteps = TARGET_TIMESTEPS - model.num_timesteps
    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps, 
            callback=callback, 
            tb_log_name="PPO-MarioBros",
            reset_num_timesteps=not os.path.exists(best_model_path)
        )
    
    final_model_path = os.path.join(permanent_log_dir, "mario_bros_model_final.zip")
    print("------------- Saving final model -------------")
    model.save(final_model_path)

    print("------------- Performing final log sync -------------")
    if os.path.exists(permanent_monitor_path): shutil.rmtree(permanent_monitor_path)
    if os.path.exists(local_monitor_dir): shutil.copytree(local_monitor_dir, permanent_monitor_path)
    if os.path.exists(permanent_board_path): shutil.rmtree(permanent_board_path)
    if os.path.exists(local_board_dir): shutil.copytree(local_board_dir, permanent_board_path)

    print("------------- Done Learning -------------")
    
    # ===== TESTING SECTION (FIXED) =====
    print("Testing trained PPO model...")
    
    # Create the base environment
    test_env = gym.make(env_id, render_mode='human')
    
    # Apply all the same wrappers as the training environment
    test_env = AtariWrapper(test_env)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecFrameStack(test_env, n_stack=4)
    
    # Load the final model for a definitive test
    final_model = PPO.load(final_model_path)
    
    obs = test_env.reset()
    
    print("Watching AI play Mario Bros...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            action, _states = final_model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            if done[0]:
                print("Episode finished. Resetting...")
                obs = test_env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping testing.")
    
    test_env.close()
    print("Testing complete!")
