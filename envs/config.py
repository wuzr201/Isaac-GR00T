from typing import OrderedDict

import cv2
import numpy as np

from kuavo_msgs.msg import sensorsData, lejuClawState
from sensor_msgs.msg import Image


def process_Image(msg, data_dict, name, ts=None):
    if msg.encoding != 'rgb8':
        raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected 'rgb8'.")

    img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    if msg.encoding == 'bgr8':
        cv_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    else:
        cv_img = img_arr
    if ts is None:
        ts = msg.header.stamp.to_sec()
    data_dict[name]['data'].append(cv_img)
    data_dict[name]['ts'].append(ts)


def process_sensorsData(msg, data_dict, name, ts=None):
    arm_begin = 4
    arm_end = 17
    if ts is None:
        ts = msg.header.stamp.to_sec()

    data = msg.joint_data.joint_q
    data = list(data[arm_begin:arm_end + 1])
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)


def process_JointState(msg, data_dict, name, ts=None):
    joint_q = np.deg2rad(msg.position)

    if ts is None:
        ts = msg.header.stamp.to_sec()

    data_dict[name]['data'].append(list(joint_q))
    data_dict[name]['ts'].append(ts)


def process_lejuClawData(msg, data_dict, name, ts=None):
    data = [
        msg.data.position[0],
        msg.data.position[1],
    ]
    if ts is None:
        ts = msg.header.stamp.to_sec()
    data_dict[name]['data'].append(data)
    data_dict[name]['ts'].append(ts)


class EnvConfig:
    img_names = [
        "cam_head",
        "cam_left",
        "cam_right"
    ]

    state_names_by_topic = OrderedDict({
        "dof_state": ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6",
                      "arm_joint_7",
                      "arm_joint_8", "arm_joint_9", "arm_joint_10", "arm_joint_11", "arm_joint_12", "arm_joint_13",
                      "arm_joint_14"],
        "leju_claw_state": ["left_claw_position", "right_claw_position"],
    })

    action_names_by_topic = OrderedDict({
        "action_arm": [
            "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6",
            "arm_joint_7",
            "arm_joint_8", "arm_joint_9", "arm_joint_10", "arm_joint_11", "arm_joint_12", "arm_joint_13",
            "arm_joint_14",
        ],
        "action_claw": [
            "left_claw_action", "right_claw_action"
        ],
    })

    img_topic_names = img_names
    obs_topic_names = [key for key in state_names_by_topic.keys()]
    action_topic_names = [key for key in action_names_by_topic.keys()]

    states_names = []
    for k, v in state_names_by_topic.items():
        states_names += v

    action_names = []
    for k, v in state_names_by_topic.items():
        action_names += v


topic_info = {
    "cam_head": {
        "topic": "/camera/color/image_raw",
        'msg_type': Image,
        'frequency': 30,
        "msg_process_fn": process_Image,
        "shape": None,
    },

    "cam_left": {
        "topic": "/left_cam/color/image_raw",
        'msg_type': Image,
        'frequency': 30,
        "msg_process_fn": process_Image,
        "shape": None,
    },

    "cam_right": {
        "topic": "/right_cam/color/image_raw",
        'msg_type': Image,
        'frequency': 30,
        "msg_process_fn": process_Image,
        "shape": None,
    },

    "dof_state": {
        "topic": "/sensors_data_raw",
        'msg_type': sensorsData,
        'frequency': 100,
        "msg_process_fn": process_sensorsData,
        "shape": None,
    },

    "leju_claw_state": {
        "topic": "/leju_claw_state",
        'msg_type': lejuClawState,
        'frequency': 500,
        "msg_process_fn": process_lejuClawData,
        "shape": None,
    },

    "action_arm": {
        "topic": "/kuavo_arm_traj",
        "msg_process_fn": process_JointState,
        'frequency': 500,
        "shape": (14,),
    },

    "action_claw": {
        "topic": "/leju_claw_command",
        "msg_process_fn": process_lejuClawData,
        'frequency': 500,
        "shape": (2,),
    },
}
