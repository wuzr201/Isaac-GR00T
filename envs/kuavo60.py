import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import time

import numpy as np
import rospy

from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from collections import deque
import math
from kuavo_msgs.msg import sensorsData, lejuClawCommand, lejuClawState
from tqdm import tqdm

from envs.config import topic_info, EnvConfig

class TargetPublisher:
    def __init__(self):
        self.arm_action_publisher = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
        self.claw_action_publisher = rospy.Publisher('/leju_claw_command', lejuClawCommand, queue_size=10)
        self.last_action_exec_time = time.time()

    def publish_target_arm_claw(self, arm_action: np.ndarray, claw_action: np.ndarray, control_arm: bool = True, control_claw: bool = True):
        """
        发布arm和夹爪的目标
        Args:
            arm_action: 手臂关节角度 (14维)
            claw_action: 夹爪位置 [left_claw, right_claw] (2维)
            control_arm: 是否控制手臂
            control_claw: 是否控制夹爪
        """
        msg_arm = JointState()
        msg_arm.header.stamp = rospy.Time.now()
        msg_arm.name = [
            "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
            "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
        ]
        msg_arm.position = np.rad2deg(arm_action.tolist())
        msg_claw = lejuClawCommand()
        msg_claw.data.name = ['left_claw', 'right_claw']
        msg_claw.data.position = claw_action
        msg_claw.data.velocity = [90.0, 90.0]
        msg_claw.data.effort = [1.0, 1.0]

        if control_arm:
            self.arm_action_publisher.publish(msg_arm)
        if control_claw:
            self.claw_action_publisher.publish(msg_claw)


ISAAC_SIM_CAMERA_FLAG = False
USE_WBC_OBS = False


class ObsBuffer:
    def __init__(self):
        
        self.img_topic_map = {}
        
        camera_names = EnvConfig.img_topic_names
        
        for camera_name in camera_names:
            if camera_name in topic_info:
                camera_config = topic_info[camera_name]
                self.img_topic_map[camera_name] = {
                    'topic': camera_config['topic'],
                    'msg_type': Image,
                    'frequency': 30,
                    'callback': self.common_callback,
                    'size_wh': (640, 480)
                }

        self.obs_topic_map = {
            'dof_state': {
                'topic': '/sensors_data_raw',
                'msg_type': sensorsData,
                'frequency': 30,
                'callback': self.common_callback,
            },
            'leju_claw_state': {
                'topic': '/leju_claw_state',
                'msg_type': lejuClawState,
                'frequency': 30,
                'callback': self.common_callback,
            },
        }

        self.base_action = None
        self.arm_action = None

        # ---------- init obs_buffer_data --------------- #
        self.obs_buffer_data = {key: {"data": deque(maxlen=self.img_topic_map[key]["frequency"]),"ts": deque(maxlen=self.img_topic_map[key]["frequency"]),} \
                                for key in self.img_topic_map}

        self.obs_buffer_data.update({key: {"data": deque(maxlen=self.obs_topic_map[key]["frequency"]),"ts": deque(maxlen=self.obs_topic_map[key]["frequency"]),} \
                                    for key in self.obs_topic_map})

        self.setup_subscribers()

    def setup_subscribers(self):
        """
        对于每个话题，创建一个ros subscriber
        Returns:

        """
        self.suber_dict = {}
        for obs_name, obs_info in self.obs_topic_map.items():
            topic = obs_info['topic']
            msg_type = obs_info['msg_type']
            frequency = obs_info['frequency']
            callback = obs_info['callback']

            suber = rospy.Subscriber(topic, msg_type, callback, callback_args=obs_name)

            print(callback)
            self.suber_dict[obs_name] = suber

        for obs_name, obs_info in self.img_topic_map.items():
            topic = obs_info['topic']
            msg_type = obs_info['msg_type']
            frequency = obs_info['frequency']
            callback = obs_info['callback']

            # 创建一个subscriber
            suber = rospy.Subscriber(topic, msg_type, callback, callback_args=obs_name)
            self.suber_dict[obs_name] = suber

    # --------- some callback functions --------------- #

    def common_callback(self, msg, name: str):
        # 检查 name 是否在 topic_info 中，如果不在则跳过处理（可能是已删除的topic）
        if name not in topic_info:
            rospy.logwarn(f"Skipping callback for topic '{name}' as it is not in topic_info (may have been removed)")
            return
        process_fn = topic_info[name]['msg_process_fn']
        process_fn(msg, self.obs_buffer_data, name)


    # ----------- 以特殊方式从buffer里获取数据 --------------- #
    def get_latest_k_state(self, k_frames_per_topic):
        """

        Args:
            k_frames_per_topic: 每个话题要取的k是多少

        Returns:

        """
        out = {}
        for name, info in self.obs_topic_map.items():
            k = k_frames_per_topic[name]
            out[name] = {
            "data": np.asarray(list(self.obs_buffer_data[name]["data"])[-k:]),  # 取最后的k个
            "robot_receive_timestamp": np.asarray(list(self.obs_buffer_data[name]["ts"])[-k:])  # 取最后的k个
            }

        return out


    def get_latest_k_img(self, k_frames_per_img_topic):
        """
        获取图像的buffer
        Args:
            k_frames_per_img_topic: 每个话题要取的k是多少

        Returns:

        """
        out = {}
        for name, info in self.img_topic_map.items():
            k = k_frames_per_img_topic[name]
            out[name] = {
                "data": np.asarray(list(self.obs_buffer_data[name]["data"])[-k:]),  # 取最后的k个
                "robot_receive_timestamp": np.asarray(list(self.obs_buffer_data[name]["ts"])[-k:])  # 取最后的k个
            }

        return out

    # ---------------- 一些启动和检查buffer的函数 ---------------- #

    def obs_buffer_is_ready(self):
        """
        所有观测初始化成功的判断
        Args:
            just_img:

        Returns:

        """
        return all([len(self.obs_buffer_data[key]["data"]) == self.img_topic_map[key]["frequency"] for key in self.img_topic_map]) and \
            all([len(self.obs_buffer_data[key]["data"]) == self.obs_topic_map[key]["frequency"] for key in self.obs_topic_map])

    def wait_buffer_ready(self, just_img: bool = False):
        progress_bars = {}
        position = 0

        for key in self.img_topic_map:
            progress_bars[key] = tqdm(
                total=self.img_topic_map[key]["frequency"],
                desc=f"Filling {key}",
                position=position,
                leave=True
            )
            position += 1

        for key in self.obs_topic_map:
            progress_bars[key] = tqdm(
                total=self.obs_topic_map[key]["frequency"],
                desc=f"Filling {key}",
                position=position,
                leave=True
            )
            position += 1

        try:
            while not self.obs_buffer_is_ready():
                for key in self.img_topic_map:
                    current_len = len(self.obs_buffer_data[key]["data"])
                    progress_bars[key].n = current_len
                    progress_bars[key].refresh()

                for key in self.obs_topic_map:
                    current_len = len(self.obs_buffer_data[key]["data"])
                    progress_bars[key].n = current_len
                    progress_bars[key].refresh()

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n[Interrupted] Exiting by user Ctrl+C.")

        print("All buffers are ready!")
        time.sleep(0.5)

class KuavoEnv:
    def __init__(self, claw_lock_threshold=50.0, claw_lock_count_threshold=5, claw_locked_value=90.0):
        # rospy.init_node('kuavo60_env', anonymous=True)
        self.target_publisher = TargetPublisher()
        self.obs_buffer = ObsBuffer()
        self.control_frequency = 100
        self.control_dt = 1.0 / self.control_frequency
        self.obs_topic_map = self.obs_buffer.obs_topic_map
        self.img_topic_map = self.obs_buffer.img_topic_map

        self.last_action_exec_time = time.time()

        self.n_obs_steps = 1

    def get_obs(self):
        """
        订阅ros话题，获取当前状态
        Returns:
        """
        k_frames_per_img_topic = {
            name: min(
                self.img_topic_map[name]["frequency"],
                math.ceil(
                    (self.n_obs_steps + 3)
                    * (self.img_topic_map[name]["frequency"] / self.control_frequency)
                ),
            )
            for name in self.img_topic_map
        }

        last_img_data = self.obs_buffer.get_latest_k_img(k_frames_per_img_topic)

        dt = self.control_dt
        timestamps = []
        for x in last_img_data.values():
            ts = x["robot_receive_timestamp"]
            if len(ts) >= 2:
                timestamps.append(ts[-2])
            elif len(ts) >= 1:
                timestamps.append(ts[-1])
            else:
                print(f"Warning: Empty timestamp array in image data")
                timestamps.append(0.0)
        
        if timestamps:
            last_timestamp = np.min(timestamps)
        else:
            print("Error: No valid timestamps found")
            last_timestamp = 0.0

        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        camera_obs_ts = dict()
        for name, value in last_img_data.items():
            topic_ts = value["robot_receive_timestamp"]
            picked_idx = list()
            for t in obs_align_timestamps:
                idx = np.argmin(np.abs(topic_ts - t))
                picked_idx.append(idx)

            camera_obs[name] = value["data"][picked_idx]
            camera_obs_ts[name] = topic_ts[picked_idx]

        k_frames_per_topic = {
                name: min(self.obs_topic_map[name]["frequency"],
                          math.ceil((self.n_obs_steps + 3) * (self.obs_topic_map[name]["frequency"] / self.control_frequency)))
            for name in self.obs_topic_map if 'img' not in name
            }
        last_robot_data = self.obs_buffer.get_latest_k_state(k_frames_per_topic)

        robot_obs = dict()
        robot_obs_ts = dict()
        for name, value in last_robot_data.items():
            topic_ts = value["robot_receive_timestamp"]
            picked_idx = list()
            for t in obs_align_timestamps:
                this_idx = np.argmin(np.abs(topic_ts - t))
                picked_idx.append(this_idx)

            robot_obs[name] = value["data"][picked_idx]
            robot_obs_ts[name] = topic_ts[picked_idx]

        obs_data = dict(camera_obs)

        all_non_img_states = np.concatenate((robot_obs['dof_state'],
                                            robot_obs['leju_claw_state']), axis=1)
        
        obs_data.update(
            {
                "state": all_non_img_states,
            }
        )

        return obs_data, robot_obs

    def exec_actions(
        self,
        actions: np.ndarray,
        control_arm: bool = True,
        control_claw: bool = True,
    ):
        """
        把网络推理出的action变成话题发布
        Args:
            actions: 动作数组
                    - ["Left_arm", "Right_arm", "Left_claw", "Right_claw"]: 16维
            control_arm: 是否控制手臂
            control_claw: 是否控制夹爪
        """
        actions = np.asarray(actions)

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        left_arm_action = actions[0, 0:7]
        right_arm_action = actions[0, 7:14]
        left_claw_action = actions[0, 14:15]
        right_claw_action = actions[0, 15:16]
            
        arm_action = np.concatenate([left_arm_action, right_arm_action])
        claw_action = np.array([left_claw_action, right_claw_action])
        
        # if claw_action is not None:
        #     claw_action = self._apply_claw_lock(claw_action)

        if arm_action is not None and claw_action is not None:
            if len(arm_action) >= 14:
                arm_action[3] = np.clip(arm_action[3], np.deg2rad(-130), np.deg2rad(0.0))
                arm_action[10] = np.clip(arm_action[10], np.deg2rad(-130), np.deg2rad(0.0))
            
            self.target_publisher.publish_target_arm_claw(
                arm_action=arm_action,
                claw_action=claw_action,
                control_arm=control_arm,
                control_claw=control_claw
            )

        dt = self.control_dt
        duration = time.time() - self.last_action_exec_time
        time_to_sleep = max(0, dt - duration)
        time.sleep(time_to_sleep)
        self.last_action_exec_time = time.time()
