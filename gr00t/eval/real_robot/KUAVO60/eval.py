import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from dataclasses import dataclass
import logging
from typing import Any, Dict
import time
import numpy as np
import torch
import threading

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy

from envs.kuavo60 import KuavoEnv
from kuavo_humanoid_sdk.kuavo_strategy_pytree.common.robot_sdk import RobotSDK

import tyro


@dataclass
class ArgsConfig:
    action_horizon: int = 8
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    model_path: str | None = None
    denoising_steps: int = 4
    modality_keys: list[str] | None = None


TASK_MAP = {
    "0" : "pick 4322 green 64", "1" : "pick 4322 green 86", "2" : "pick 4322 green 106",
    "3" : "unpack left 4322 green 64", "4" : "unpack left 4322 green 86", "5" : "unpack left 4322 green 106",
    "6" : "unpack right 4322 green 64", "7" : "unpack right 4322 green 86", "8" : "unpack right 4322 green 106",
    "9" : "pick 4611 green 64", "10" : "pick 4611 green 86", "11" : "pick 4611 green 106",
    "12" : "unpack left 4611 green 64", "13" : "unpack left 4611 green 86", "14" : "unpack left 4611 green 106",
    "15" : "unpack right 4611 green 64", "16" : "unpack right 4611 green 86", "17" : "unpack right 4611 green 106",
    "18" : "pick 4633 green 64", "19" : "pick 4633 green 86", "20" : "pick 4633 green 106",
    "21" : "unpack left 4633 green 64", "22" : "unpack left 4633 green 86", "23" : "unpack left 4633 green 106",
    "24" : "unpack right 4633 green 64", "25" : "unpack right 4633 green 86", "26" : "unpack right 4633 green 106",
}

class EpisodeManager:
    def __init__(self):

        self.episode_active = False
        self.waiting_for_result = False

        self.success = 0
        self.failure = 0
        self.episode_idx = 0
        self._exit = False

        self._lock = threading.Lock()
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._result_event = threading.Event()

        self._result = None
        self._exit = False

        self._listener_thread = threading.Thread(
            target=self._keyboard_listener, daemon=True
        )
        self._listener_thread.start()

        self.task = None

    def _keyboard_listener(self):
        while True:
            key = input().strip().lower()

            with self._lock:
                # ========== GLOBAL ==========
                if key == "q":
                    print("\n[EXIT] Quit requested")
                    self._exit = True
                    self._stop_event.set()
                    self._start_event.set()
                    self._result_event.set()
                    return

                # ========== IDLE ==========
                if not self.episode_active and not self.waiting_for_result:
                    print(f"\nTASK_MAP: {TASK_MAP}")
                    print(f"\nPick A TASK: {TASK_MAP.keys()} and to START episode {self.episode_idx}")
                    if key in TASK_MAP.keys():
                        self.task = TASK_MAP[key]
                        self.episode_active = True
                        self.episode_idx += 1
                        print(f"\nEpisode {self.episode_idx} START")
                        self._start_event.set()
                    continue

                # ========== RUNNING ==========
                if self.episode_active and not self.waiting_for_result:
                    if key == "s":
                        print("\nEpisode STOP")
                        self.episode_active = False
                        self._stop_event.set()
                    continue

                if self.waiting_for_result:
                    if key in ["y", "n"]:
                        self._result = key
                        self._result_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set() or self._exit

    def end_episode_and_wait_result(self):
        if self._exit:
            raise KeyboardInterrupt

        self.waiting_for_result = True
        print("Episode finished. Input [y/n] to mark success:")

        self._result_event.clear()
        self._result_event.wait()

        if self._result == "y":
            self.success += 1
            print("✔ SUCCESS")
        else:
            self.failure += 1
            print("✘ FAILURE")

        total = self.success + self.failure
        print(
            f"Success rate: {self.success}/{total} = {self.success / total:.2%}"
        )

        # 回合结束，重置状态
        time.sleep(0.1)  # 延迟避免回车丢失
        self.waiting_for_result = False
        self._result = None
        self.episode_active = False
        self._start_event.clear()
        self._stop_event.clear()



def recursive_add_extra_dim(obs: Dict) -> Dict:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


def obs_to_policy_inputs(obs: Dict[str, Any], modality_config) -> Dict:
    model_obs = {}

    model_obs["video"] = {
        k: obs[k] for k in modality_config["video"].modality_keys
    }

    state = np.array(obs["state"], dtype=np.float32)
    model_obs["state"] = {
        "left_arm": state[:,:7],
        "right_arm": state[:,7:14],
        "gripper": state[:,14:16],
    }

    model_obs["language"] = {
        "annotation.human.task_description": [obs["lang"]]
    }

    model_obs = recursive_add_extra_dim(model_obs)

    return model_obs


def resample_action_chunk(action_chunk, source_dt=0.1, target_dt=0.01):
    action_chunk = np.asarray(action_chunk)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)

    if action_chunk.shape[0] <= 1:
        return action_chunk

    total_duration = source_dt * (action_chunk.shape[0] - 1)
    num_target_steps = int(round(total_duration / target_dt)) + 1

    source_times = np.linspace(0, total_duration, action_chunk.shape[0])
    target_times = np.linspace(0, total_duration, num_target_steps)

    out = np.zeros((num_target_steps, action_chunk.shape[1]))
    for d in range(action_chunk.shape[1]):
        out[:, d] = np.interp(target_times, source_times, action_chunk[:, d])
    return out


def resample_chunk_with_claw_hold(
    action_chunk,
    previous_action,
    control_frequency,
    source_dt=0.1,
    arm_dims=slice(0, 14),
    claw_dims=slice(14, 16),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    action_chunk = np.asarray(action_chunk)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)

    if previous_action is not None:
        source = np.vstack([previous_action, action_chunk])
        resampled = resample_action_chunk(
            source, source_dt, 1.0 / control_frequency
        )[1:]
    else:
        source = action_chunk
        resampled = resample_action_chunk(
            action_chunk, source_dt, 1.0 / control_frequency
        )

    total_duration = source_dt * max(source.shape[0] - 1, 1)
    target_times = np.linspace(0, total_duration, resampled.shape[0])
    source_times = np.linspace(0, total_duration, source.shape[0])
    hold_idx = np.searchsorted(source_times, target_times, side="right") - 1
    hold_idx = np.clip(hold_idx, 0, source.shape[0] - 1)

    resampled[:, claw_dims] = source[hold_idx][:, claw_dims]

    return torch.from_numpy(resampled).to(device)


def reset_arm_pose(env: KuavoEnv):
    obs, robot_obs = env.get_obs()
    current_arm_state = obs["state"][0][:14]  # 当前手臂位置
    current_claw_state = robot_obs['leju_claw_state']

    open_claw_value = np.zeros([2]).astype(float)  # [0.0, 0.0]
    open_claw_action = np.concatenate([current_arm_state, open_claw_value])
    env.exec_actions(actions=open_claw_action, control_arm=False, control_claw=True)
    time.sleep(1)
    
    current_claw_state = open_claw_value.copy()
    obs_data, robot_obs= env.get_obs()

    init_arm_joint = np.array([-0.13108279410748547, 0.8129094913924698, -0.24350018506614143, -2.417029590409915, 1.3373353902132268, 0.40744714341534344, 0.9697406158372316,
                               -0.13080062822719687, -0.8167237074915805, 0.2473180179648636, -2.410544954816824,-1.3350419244819163,-0.40477685656131995, 0.9716470019773075 ])
    current_joints = obs_data["state"][0][:14]
    dt = 0.1

    num_points = 30

    # 从current_joints到init_joints插值
    for i in range(1, num_points + 1):
        # 线性插值
        alpha = i / num_points
        interp_joints = current_joints + (init_arm_joint - current_joints) * alpha
        
        action = np.concatenate([interp_joints, current_claw_state])
        
        # 使用exec_actions执行动作
        env.exec_actions(actions=action, control_arm=True, control_claw=True)
        time.sleep(dt)
    
    logging.info("Arm reset completed!")


class EvaluationThread(threading.Thread):
    def __init__(self, policy, env, cfg, episode_mgr):
        super().__init__(daemon=True)
        self.policy = policy
        self.env = env
        self.cfg = cfg
        self.episode_mgr = episode_mgr

    def run(self):
        while not self.episode_mgr._exit:
            # 等待 START
            while not self.episode_mgr.episode_active and not self.episode_mgr._exit:
                time.sleep(0.01)

            if self.episode_mgr._exit:
                break

            last_action = None
            print("Inference running...")

            while self.episode_mgr.episode_active and not self.episode_mgr._exit:
                obs, _ = self.env.get_obs()
                obs["lang"] = self.episode_mgr.task

                model_input = obs_to_policy_inputs(
                    obs, self.policy.get_modality_config()
                )
                action_dict = self.policy.get_action(model_input)[0]

                # 拼接动作
                left_arm = action_dict["left_arm"]
                right_arm = action_dict["right_arm"]
                gripper = action_dict["gripper"]
                action_chunk = np.concatenate([left_arm, right_arm, gripper], axis=2)[0]

                actions = action_chunk[:self.cfg.action_horizon, :]
                if last_action is None:
                    current_arm_state = obs["state"][0][:14]
                    current_claw_state = np.array([0.0, 0.0])
                    last_action = np.concatenate([current_arm_state, current_claw_state], axis=0)

                resampled = resample_chunk_with_claw_hold(
                    actions,
                    previous_action=last_action,
                    control_frequency=100.0,
                )

                last_action = actions[-1, :]

                for a in resampled:
                    if self.episode_mgr.should_stop():
                        break
                    self.env.exec_actions(a.cpu().numpy())

            # 推理循环结束，等待用户标记结果
            self.episode_mgr.end_episode_and_wait_result()
            self.policy.reset()
            reset_arm_pose(self.env)


def main(args: ArgsConfig):
    logging.basicConfig(level=logging.INFO)

    if args.model_path is None:
        raise ValueError("Model path is required")

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    robot_sdk = RobotSDK()
    robot_sdk.control.set_external_control_arm_mode()
    robot_sdk.control.control_head(0, np.deg2rad(20))

    env = KuavoEnv()
    env.obs_buffer.wait_buffer_ready()

    reset_arm_pose(env)
    robot_sdk.control.set_arm_quick_mode(True)

    episode_mgr = EpisodeManager()
    policy.reset()

    eval_thread = EvaluationThread(policy, env, args, episode_mgr)
    eval_thread.start()

    while not episode_mgr._exit:
        time.sleep(0.1)

if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)