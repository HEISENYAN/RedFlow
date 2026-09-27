# import packages and module here
import numpy as np
import torch
import dill
import os, sys
import copy
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *

from envs._base_task import Base_Task
from typing import Dict, List, Optional, Tuple
def encode_obs(observation):  # Post-Process Observation

    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    # ...
    return input_rgb_arr, input_state


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    train_config_name, model_name, checkpoint_id, pi0_step ,checkpoint_base_dir= (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"], usr_args["checkpoint_base_dir"])
    return PI0(train_config_name, model_name, checkpoint_id, checkpoint_base_dir, pi0_step)

def eval_one_step(task_env: Base_Task, model: PI0, eval_config: Dict, verbose: bool = False):
    """
    Standalone eval_one_step function wrapper for PI0 model
    Matches the interface pattern from RoboTwinEvalModel
    """
    lang_rsp = model.eval_one_step(task_env, eval_config, verbose)
    return lang_rsp

def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========
    data = []

    actions = model.get_action()[:model.pi0_step]

    
    for action in actions:
        # Create a new data_point dict for each iteration
        data_point = {
            "observation": copy.deepcopy(model.observation_window),
            "action": action,
            "reward": TASK_ENV.eval_success,
            "timestep": TASK_ENV.take_action_cnt
        }
        data.append(data_point)
        
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        if TASK_ENV.eval_success:
            return data
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        
    
    return data
    # ============================


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()

def reset(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
