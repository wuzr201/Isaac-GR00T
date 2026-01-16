import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dataclasses
from pathlib import Path
import shutil

from typing import Literal
import os
import rosbag
from collections import defaultdict

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import torch
import tqdm
import argparse

from envs.config import topic_info, EnvConfig


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class DataConverter:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.topic_data = defaultdict(dict)

    def _copy_data_and_set_zero(self, data, ts):
        """
        把数据拷贝并且设置为0
        Args:
            data_dict:

        Returns:

        """
        return np.zeros_like(data)

    def forward_fill_on_grid(self, msg_dict, time_grid, shape=None):
        """
        对一个 topic 的消息进行前向填充，按 time_grid 对齐

        参数：
        - msg_list: List of (timestamp, message)，按原始时间排序的消息列表
        - time_grid: 均匀的时间戳序列，例如 np.arange(...)

        返回：
        - 一个列表，与 time_grid 等长，每个位置是最近的消息（或 None）
        """
        msg_list = sorted(
            zip(msg_dict["ts"], msg_dict["data"]), key=lambda x: x[0]
        )
        result_list = []
        idx = 0

        current_msg = None

        for t in time_grid:
            if current_msg is None:
                current_msg = (
                    self._copy_data_and_set_zero(msg_list[0][1], t)
                    if msg_list
                    else list(np.zeros(shape))
                )

            while idx < len(msg_list) and msg_list[idx][0] <= t:
                current_msg = msg_list[idx][1]
                idx += 1
            result_list.append(current_msg)

        return result_list

    def depth_to_pointcloud(self, depth_images, intrinsic_matrices, num_points=512):
        """
        Convert depth image(s) to point cloud(s) using pinhole camera model.

        Parameters:
        - depth_images: np.ndarray
            Shape (H, W) for single image or (T, H, W) for multiple images
        - intrinsic_matrices: np.ndarray
            Shape (3, 3) for single camera or (T, 3, 3) for per-frame intrinsics
        - num_points: int
            Number of points to sample per point cloud

        Returns:
        - np.ndarray
            (num_points, 3) for single image input
            (T, num_points, 3) for sequence input
        """

        def _single_depth_to_pc(depth_image, intrinsic_matrix):
            H, W = depth_image.shape
            fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
            cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

            u, v = np.meshgrid(np.arange(W), np.arange(H))

            valid_mask = (depth_image > 0) & np.isfinite(depth_image)
            Z = depth_image[valid_mask]
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            X = (u_valid - cx) * Z / fx
            Y = (v_valid - cy) * Z / fy

            points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
            points = points[Z.reshape(-1) > 0]

            if len(points) == 0:
                return np.zeros((num_points, 3), dtype=np.float32)

            idxs = np.random.choice(len(points), num_points, replace=len(points) < num_points)
            return points[idxs].astype(np.float32)

        depth_images = np.array(depth_images)
        intrinsic_matrices = np.array(intrinsic_matrices)

        if depth_images.ndim == 2:
            return _single_depth_to_pc(depth_images, intrinsic_matrices)

        pcs = [
            _single_depth_to_pc(
                depth_images[t][:, :, 0],
                intrinsic_matrices[t] if intrinsic_matrices.ndim == 3 else intrinsic_matrices
            )
            for t in range(depth_images.shape[0])
        ]
        return np.stack(pcs, axis=0)

    def process_rosbag(self, bag_path, dt=0.1):
        bag = rosbag.Bag(bag_path)
        topic_data = defaultdict(dict)
        print(f'==== start processing bag {bag_path}')

        names = []

        all_topics_min_max_time = {}

        topic_name_to_nick_name = {}
        for name, info in topic_info.items():
            topic_name_to_nick_name[info["topic"]] = name

        for topic, msg, t in bag.read_messages():
            if topic not in topic_name_to_nick_name.keys():
                continue
            name = topic_name_to_nick_name[topic]
            info = topic_info[name]
            msg_process_fn = info["msg_process_fn"]

            if name not in topic_data.keys():
                topic_data[name] = defaultdict(list)
                names.append(name)

                all_topics_min_max_time[name] = [float("inf"), float("-inf")]

            ts = t.to_sec()

            all_topics_min_max_time[name][0] = min(all_topics_min_max_time[name][0], ts)
            all_topics_min_max_time[name][1] = max(all_topics_min_max_time[name][1], ts)

            msg_process_fn(msg, topic_data, name, ts)

        bag.close()

        for name in EnvConfig.img_topic_names + EnvConfig.obs_topic_names + EnvConfig.action_topic_names:
            assert name in topic_data.keys(), f'ERROR: target topic {name} not in bag {bag_path}'

        all_images_min_time = []
        all_images_max_time = []
        for name in EnvConfig.img_topic_names:
            all_images_min_time.append(all_topics_min_max_time[name][0])
            all_images_max_time.append(all_topics_min_max_time[name][1])
        for name in ['action_arm']:
            all_images_min_time.append(all_topics_min_max_time[name][0])
            all_images_max_time.append(all_topics_min_max_time[name][1])

        min_time = max(all_images_min_time)
        max_time = min(all_images_max_time)

        time_grid = np.arange(min_time, max_time + dt, dt)

        aligned_dict = {}
        print(f'======= processing bag: {bag_path} =======')
        for name in names:
            print(f'======= processing topic: {name} =======')
            aligned_dict[name] = self.forward_fill_on_grid(topic_data.get(name, []), time_grid,
                                                           topic_info[name]['shape'])

        print(f'==== end processing bag {bag_path}')
        return aligned_dict, time_grid

    def create_empty_dataset(
            self,
            repo_id: str,
            robot_type: str,
            mode: Literal["video", "image"] = "video",
            *,
            has_velocity: bool = False,
            has_effort: bool = False,
            dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
            root: str,
    ) -> LeRobotDataset:

        features = {}

        for img_name in EnvConfig.img_topic_names:
            if topic_info[img_name]["shape"] is not None:
                img_shape = topic_info[img_name]["shape"]
            else:
                img_shape = (3, 480, 640)  # default shape
            features[f"observation.images.{img_name}"] = {
                "dtype": mode,
                "shape": img_shape,
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

        features.update({
            "observation.state": {
                "dtype": "float32",
                "shape": (len(EnvConfig.states_names),),
                "names": {
                    "motors": EnvConfig.states_names
                }
            },
            "action": {
                "dtype": "float32",
                "shape": (len(EnvConfig.action_names),),
                "names": {
                    "motors": EnvConfig.action_names
                }
            },
        })

        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=10,
            features=features,
            robot_type=robot_type,
            use_videos=True,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
            root=root,
        )

    def load_raw_episode_data(self, ep_path: Path):
        bag_data, _ = self.process_rosbag(ep_path, dt=0.1)

        all_states = []

        for state_name in EnvConfig.obs_topic_names:
            # print("state_name:", state_name)
            this_state = np.array([data for data in bag_data[state_name]], dtype=np.float32)
            print("state_name:", state_name, "shape:", this_state.shape)
            all_states.append(this_state)

        state = np.concatenate(
            all_states, axis=1
        )

        all_actions = []
        for action_name in EnvConfig.action_topic_names:
            if action_name in bag_data.keys():

                this_action = np.array(
                    [data for data in bag_data[action_name]], dtype=np.float32
                )
            else:
                assert False, f'ERROR: action {action_name} not in bag!'
                action_shape = topic_info[action_name]["shape"]
                this_action = np.zeros([np.shape(list(bag_data.values())[0])[0], action_shape[0]])

            all_actions.append(this_action)

        action = np.concatenate(all_actions, axis=1)

        images_dict = {}
        for img_name in EnvConfig.img_topic_names:
            images_dict[img_name] = np.asarray([data for data in bag_data[img_name]])
        env_state = None
        return images_dict, env_state, state, action

    def generate_cut_start_end_idx(self, actions: np.ndarray, state: np.ndarray):
        """
        根据actions来判断生成的
        """
        valid_start = 0
        valid_end = state.shape[0]
        return valid_start, valid_end

    def populate_dataset(
            self,
            dataset: LeRobotDataset,
            bag_files: list[Path],
            task: str,
            episodes: list[int] | None = None,
            dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ) -> LeRobotDataset:
        """
        把从bag里读的数据填充到lerobot dataset里
        Args:
            dataset:
            bag_files:
            task:
            episodes:

        Returns:

        """
        all_bag_dirs = []
        for file in bag_files:
            dir_name = str(file.parent)
            if dir_name not in all_bag_dirs:
                all_bag_dirs.append(dir_name)

        if episodes is None:
            episodes = range(len(bag_files))

        for ep_idx in tqdm.tqdm(episodes):
            ep_path = bag_files[ep_idx]

            # fetch all data by default
            images_dict, env_state, state, action = self.load_raw_episode_data(ep_path)
            print("state shape:", state[0, :].shape)

            valid_start, valid_end = self.generate_cut_start_end_idx(action, state)

            for i in range(valid_start, valid_end):
                frame = {
                    "observation.state": torch.from_numpy(state[i, :]).type(torch.float32),
                    "action": torch.from_numpy(action[i, :]).type(torch.float32),
                }
                for img_name in EnvConfig.img_topic_names:
                    frame[f"observation.images.{img_name}"] = torch.from_numpy(images_dict[img_name][i, :])

                dataset.add_frame(frame, task=task)
            dataset.save_episode()

        return dataset


def cvt_bag2lerobot(
        source_dir: Path,
        source_dirs: list[Path],
        target_dir: Path,
        repo_id: str,
        task: str = None,
        *,
        episodes: list[int] | None = None,
        mode: Literal["video", "image"] = "video",
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
        # root: str,
):
    updated_dataset_config = DatasetConfig(
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

    data_converter = DataConverter(source_dir)

    dirname = os.path.basename(source_dir)
    target_dir = os.path.join(target_dir, f'mm_{dirname}')
    task_info = dirname.replace("_", " ")

    bag_files = [Path(root) / file
                    for root, _, files in os.walk(source_dir, followlinks=True)
                    for file in files if file.endswith(".bag")]

    for root, _, files in os.walk(source_dir):
        print(f'========== source_dir：{root}, files {files}')

    print(f'========== 待转换bag文件：{bag_files}')

    dataset = data_converter.create_empty_dataset(
        repo_id,
        robot_type="kuavo60",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=updated_dataset_config,
        root=target_dir,
    )

    data_converter.populate_dataset(
        dataset,
        bag_files,
        task=task_info,
        episodes=episodes,
        dataset_config=updated_dataset_config,
    )

    images_dir = Path(target_dir) / "images"
    if images_dir.exists():
        print(f"Removing images folder: {images_dir}")
        shutil.rmtree(images_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert rosbag to LeRobot dataset')

    source_dirs = [
        '', # replace with your path of rosbag to be converted
    ]

    for source_dir in source_dirs:
        try:
            cvt_bag2lerobot(
                source_dir=source_dir,
                source_dirs=None,
                target_dir='/mnt/ssd/datasets/kuavo60_lerobot/', # replace with your path of target dataset
                repo_id=0,
            )
        except AssertionError:
            
            continue
