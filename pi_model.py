#!/usr/bin/env python3
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from envs._base_task import Base_Task
from typing import Dict, List, Optional, Tuple
class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, checkpoint_base_dir, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir=f"{checkpoint_base_dir}/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}")
            #checkpoint_dir=download.maybe_download("gs://openpi-assets/checkpoints/pi0_base"))
            #robotwin_repo_id=model_name)
        #self.policy.eval()
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    
    def encode_obs(self, observation, save_dir, step):  # Post-Process Observation

        input_rgb_arr = [
            observation["observation"]["head_camera"]["rgb"],
            observation["observation"]["right_camera"]["rgb"],
            observation["observation"]["left_camera"]["rgb"],
        ]
        input_state = observation["joint_action"]["vector"]
        # ...

        import cv2
        import os
        img_dict = {}
        for cam_name in ["head_camera", "left_camera", "right_camera", "front_camera"]:
            if cam_name in observation["observation"]:
                img_dict[cam_name.replace("_camera", "_image")] = observation["observation"][cam_name]["rgb"]
        resize_size = (320, 240)
        if resize_size is not None:
            for name, img in img_dict.items():
                img_dict[name] = cv2.resize(img, resize_size)
        
        if save_dir is not None and step is not None:
            for name, img in img_dict.items():
                sub_dir = os.path.join(save_dir, name)
                os.makedirs(sub_dir, exist_ok=True)
                out_path = os.path.join(sub_dir, f"{name}_step_{step}.png")
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return input_rgb_arr, input_state
    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
    def eval_one_step(self, task_env: Base_Task, eval_config: Dict, verbose: bool = False):
        """
        Execute one evaluation step for PI0 model
        All the function interfaces below are just examples
        You can modify them according to your implementation
        But we strongly recommend keeping the code logic unchanged
        """
        import time
        
        if verbose:
            start_total = time.perf_counter()
        
        # Get current observation from environment
        observation = task_env.get_obs()
        
        # Set language instruction if not already set (first step initialization)
        if self.observation_window is None:
            instruction = task_env.get_instruction()
            self.set_language(instruction)
        
        # Encode observation for PI0 model
        input_rgb_arr, input_state = self.encode_obs(observation, save_dir=eval_config["tmp_video_save_dir"], step=task_env.take_action_cnt)
        
        # Update observation window with current observation
        self.update_observation_window(input_rgb_arr, input_state)
        
        if verbose:
            start_query_action = time.perf_counter()
        
        # Get action from model with retry mechanism
        try_time = 0
        action_success = False
        actions = None
        
        while try_time < 10:
            try:
                # Get actions and slice to pi0_step as per original eval logic
                actions = self.get_action()[:self.pi0_step]
                action_success = True
                break
            except Exception as e:
                if verbose:
                    print(f"[Warning] Failed to get action, retry {try_time + 1}/10: {e}")
                time.sleep(1)
                try_time += 1
                continue
        
        if not action_success or actions is None:
            print("[Error] PI0 model failed to generate action after 10 retries.")
            return None
        
        if verbose:
            end_query_action = time.perf_counter()
            elapsed_ms_query_action = (end_query_action - start_query_action) * 1000
            print(f"Query Action time consuming: {elapsed_ms_query_action:.3f} ms")
        
        # Execute action in environment
        if verbose:
            start_take_action = time.perf_counter()
        
        # PI0 outputs actions, execute them sequentially
        lang_rsp = "Not None"  # PI0 does not return language response like LLM-based models
        
        # Execute each action in the predicted sequence
        for action in actions:
            task_env.take_action(action)
            # Check if task is completed successfully
            if task_env.eval_success:
                break
            # Update observation for next action if there are more actions to execute
            if len(actions) > 1:
                observation = task_env.get_obs()
                input_rgb_arr, input_state = self.encode_obs(observation, save_dir=eval_config["tmp_video_save_dir"], step=task_env.take_action_cnt)
                self.update_observation_window(input_rgb_arr, input_state)
        
        if verbose:
            end_take_action = time.perf_counter()
            elapsed_ms_take_action = (end_take_action - start_take_action) * 1000
            print(f"Take Action time cost: {elapsed_ms_take_action:.3f} ms")
            
            end_total = time.perf_counter()
            elapsed_ms_total = (end_total - start_total) * 1000
            print(f"Total time cost: {elapsed_ms_total:.3f} ms\n")
        
        return lang_rsp