import gymnasium as gym
import numpy as np
# This import seems to be from a custom file, ensure RandomAgent.py is in the same directory.
# from RandomAgent import TimeLimitWrapper 
import os
import shutil # Import the shutil library for file operations

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
# --- FIX: Import the specific error we need to catch ---
from stable_baselines3.common.monitor import LoadMonitorResultsError
import ale_py

class SaveAndSyncCallback(BaseCallback):
    """
    A callback that saves the best model to Drive and periodically syncs
    local log folders to a permanent location in Drive.
    """
    def __init__(self, check_freq: int, log_dir: str, board_dir: str, drive_log_dir: str, sync_freq: int = 10000, verbose: int = 1):
        super(SaveAndSyncCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.sync_freq = sync_freq
        self.log_dir = log_dir # Local monitor log path (e.g., "tmp/")
        self.board_dir = board_dir # Local tensorboard log path (e.g., "board/")
        self.drive_log_dir = drive_log_dir # Permanent Drive path
        
        # Paths for saving models and syncing logs to Drive
        self.best_model_save_path = os.path.join(self.drive_log_dir, 'best_model.zip')
        self.drive_monitor_path = os.path.join(self.drive_log_dir, 'monitor_logs')
        self.drive_board_path = os.path.join(self.drive_log_dir, 'board_logs')
        
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create all necessary directories in Google Drive
        os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)
        os.makedirs(self.drive_monitor_path, exist_ok=True)
        os.makedirs(self.drive_board_path, exist_ok=True)

    def _on_step(self) -> bool:
        # --- Logic to save the best model (checks every `check_freq` steps) ---
        if self.n_calls % self.check_freq == 0:
            # --- FIX: Wrap the file reading in a try...except block ---
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        # This print is now less frequent as it's tied to finding a log file
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.best_model_save_path}")
                        self.model.save(self.best_model_save_path)
            except (LoadMonitorResultsError, FileNotFoundError):
                if self.verbose > 0:
                    # This message will now only appear at the very beginning of training
                    print(f"Could not find log file at {self.log_dir}, skipping check.")
        
        # --- Logic to sync local logs to Drive (runs every `sync_freq` steps) ---
        if self.n_calls % self.sync_freq == 0:
            if self.verbose > 0:
                print(f"Syncing local logs to {self.drive_log_dir}")
            
            # Copy monitor logs
            if os.path.exists(self.drive_monitor_path):
                shutil.rmtree(self.drive_monitor_path)
            # --- FIX: Add a check to ensure the source directory exists before copying ---
            if os.path.exists(self.log_dir):
                shutil.copytree(self.log_dir, self.drive_monitor_path)

            # Copy tensorboard logs
            if os.path.exists(self.drive_board_path):
                shutil.rmtree(self.drive_board_path)
            if os.path.exists(self.board_dir):
                shutil.copytree(self.board_dir, self.drive_board_path)

        return True

if __name__ == '__main__':
    # --- DEFINE ALL PATHS ---
    local_monitor_dir = "tmp/"
    local_board_dir = "board/"
    drive_log_dir = "/content/drive/MyDrive/mario_training_synced/"
    
    # --- RESTORE LOGS IF RESUMING ---
    drive_monitor_path = os.path.join(drive_log_dir, 'monitor_logs')
    drive_board_path = os.path.join(drive_log_dir, 'board_logs')
    if os.path.exists(drive_monitor_path):
        print("Found old monitor logs in Drive, restoring to local session...")
        shutil.copytree(drive_monitor_path, local_monitor_dir, dirs_exist_ok=True)
    if os.path.exists(drive_board_path):
        print("Found old TensorBoard logs in Drive, restoring to local session...")
        shutil.copytree(drive_board_path, local_board_dir, dirs_exist_ok=True)

    env_id = 'ALE/MarioBros-v5'
    num_cpu = 4
    
    # ==============================================================================
    # === FIX APPLIED HERE =========================================================
    # ==============================================================================
    # By passing `monitor_dir`, `make_atari_env` uses the correct `Monitor` wrapper
    # that creates the .csv log files your callback needs to read.
    env = make_atari_env(env_id, n_envs=num_cpu, seed=0, monitor_dir=local_monitor_dir, vec_env_cls=DummyVecEnv)
    
    env = VecFrameStack(env, n_stack=4)
    
    # This line is now removed because the monitoring is handled correctly above.
    # Using VecMonitor was the source of the issue, as it doesn't create the .csv file.
    # env = VecMonitor(env, local_monitor_dir) 
    # ==============================================================================
    # ==============================================================================
    

    # --- LOGIC TO RESUME TRAINING ---
    best_model_path = os.path.join(drive_log_dir, "best_model.zip")
    
    if os.path.exists(best_model_path):
        print("------------- Found best model in Drive. Resuming training. -------------")
        # When loading, SB3 will automatically wrap the env with VecMonitor for console logging
        model = PPO.load(best_model_path, env=env)
    else:
        print("------------- No model found. Starting new training. -------------")
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=local_board_dir, learning_rate=0.00003)
    
    print("------------- Start Learning -------------")
    callback = SaveAndSyncCallback(check_freq=1000, log_dir=local_monitor_dir, board_dir=local_board_dir, drive_log_dir=drive_log_dir)
    
    model.learn(
        total_timesteps=5000000, 
        callback=callback, 
        tb_log_name="PPO-MarioBros",
        reset_num_timesteps=not os.path.exists(best_model_path)
    )
    
    # --- SAVE FINAL MODEL AND SYNC FINAL LOGS ---
    final_model_path = os.path.join(drive_log_dir, "mario_bros_model_final.zip")
    print("------------- Saving final model to Drive -------------")
    model.save(final_model_path)

    print("------------- Performing final log sync to Drive -------------")
    if os.path.exists(drive_monitor_path): shutil.rmtree(drive_monitor_path)
    if os.path.exists(local_monitor_dir): shutil.copytree(local_monitor_dir, drive_monitor_path)
    if os.path.exists(drive_board_path): shutil.rmtree(drive_board_path)
    if os.path.exists(local_board_dir): shutil.copytree(local_board_dir, drive_board_path)

    print("------------- Done Learning -------------")
    
    # ===== TESTING SECTION (No changes needed) =====
    print("Testing trained model...")
    test_env = gym.make(env_id, render_mode='human')
    test_env = AtariWrapper(test_env)
    obs, info = test_env.reset()
    
    print("Watching AI play Mario Bros...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, info = test_env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping testing.")
    
    test_env.close()
    print("Testing complete!")