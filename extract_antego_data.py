#!/usr/bin/env python3
"""
Extract root orientation and translation from Nymeria dataset sequences.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.sensor_data import TimeDomain
from nymeria.data_provider import NymeriaDataProvider
from nymeria.body_motion_provider import BodyDataProvider
from nymeria.xsens_constants import XSensConstants

def extract_antego_data(
    sequence_folder: Path,      # ex: PosixPath('dataset_nymeria/nymeria_firstdownload/20231211_s1_seth_bowman_act4_9jyykj')
    frame_rate: float = 30.0
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Extract root transforms, joint transforms, foot contacts, and egoview RGB image for each frame from a Nymeria sequence in Aria world coordinates.
    
    Args:
        sequence_folder: Path to the sequence folder containing the Nymeria data
        frame_rate: Desired sampling rate in fps (default: 30.0)
    
    Returns:
        Dictionary containing the following keys: (timestamp_ns, root_translation, root_orientation, joint_translation, joint_orientation, contact-information, egoview-RGB)
        - timestamp_ns: list of timestamps in nanoseconds
        - root_translation: list of (3, ) nparray representing root position
        - root_orientation: list of (3, 3) nparray representing root orientation
        - joint_translation: list of (22, 3) nparray representing joint positions
        - joint_orientation: list of (22, 3, 3) nparray representing joint orientations
        - contact_information: list of (4, ) nparray representing foot contact states (4 contact points)
        - egoview_RGB: list of (3, 1408, 1408) nparray representing egoview RGB image
    """
    # Initialize data provider
    config = {
        'sequence_rootdir': sequence_folder,
        'load_head': True,  # Loads the head pose data
        'load_body': True,
        'load_observer': False,
        'load_wrist': False,
        'trajectory_sample_fps': frame_rate
    }
    
    try:
        data_provider = NymeriaDataProvider(**config)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize data provider: {e}")
    
    if data_provider.body_dp is None:
        raise RuntimeError("No body motion data found in sequence")
    
    # Get the time span for the sequence
    timespan_ns = data_provider.timespan_ns
    start_time_ns, end_time_ns = timespan_ns
    
    # Calculate frame timestamps based on desired frame rate
    duration_ns = end_time_ns - start_time_ns   # 
    frame_interval_ns = int(1e9 / frame_rate)   # nanoseconds per frame
    
    frame_timestamps = []
    current_time = start_time_ns
    while current_time <= end_time_ns:
        frame_timestamps.append(current_time)
        current_time += frame_interval_ns
    
    antego_data = {
        'timestamp_ns': [],
        'root_translation': [],
        'root_orientation': [],
        'joint_translation': [],
        'joint_orientation': [],
        'contact_information': [],
        'egoview_RGB': [],
    }
    for timestamp_ns in frame_timestamps:
        
        try:
            # Get synced poses for this timestamp
            poses = data_provider.get_synced_poses(timestamp_ns)
            # poses is a dictionary with keys: 'recording_head', 'xsens', 'momentum'; here is the structure:
            # poses = {
            #     'recording_head': ClosedLoopTrajectoryPose,
            #     'xsens':          np.ndarray (22, 2, 3) - 
            #     'momentum':       torch.Tensor (N, 3) -(N vertices, 3 dimensions),
            # }
            
            if 'xsens' not in poses:
                continue
            # The xsens skeleton data is in format (22, 2, 3) representing bones as
            # [child_position, parent_position] pairs. To get proper root orientation
            # and translation, we need to access the raw SE3 transforms directly.

            timestamp_us = int(timestamp_ns / 1e3)

            if data_provider.recording_head is not None and 'recording_head' in poses: # If no head pose available, skip this frame
                ########################################################
                ####  Extract root transform in world coordinates ######
                ########################################################
                # Get head pose in world coordinates
                head_pose = poses['recording_head']             # head_pose is a mps.ClosedLoopTrajectoryPose object
                T_Wd_Hd = head_pose.transform_world_device      # sophus.SE3 object - from device to world coordinates, see https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory

                # Get XSens head to Aria head device transform
                T_Hd_Hx = data_provider.T_Hd_Hx(timestamp_ns)   # sophus.SE3 object - from XSens head to Aria head device
                T_Wd_Hx = T_Wd_Hd @ T_Hd_Hx                     # sophus.SE3 object - from XSens head to unified world coordinates

                # Now let's get the raw XSens data and apply world transform
                idx = data_provider.body_dp._BodyDataProvider__get_closest_timestamp_idx(timestamp_us)
                q = data_provider.body_dp.xsens_data[XSensConstants.k_part_qWXYZ][idx]  # nparray (92, ) - 23x4
                t = data_provider.body_dp.xsens_data[XSensConstants.k_part_tXYZ][idx]   # nparray (69, ) - 23x3
                T_Wx_Px = BodyDataProvider.qt_to_se3(q, t) # sophus.SE3 object - from point_i to world coordinates

                # Apply world alignment
                head_idx = XSensConstants.part_names.index("Head")      # Get the head joint index (Should be 6)
                pelvis_idx = XSensConstants.part_names.index("Pelvis")  # Get the root/pelvis joint index (Should be 0)
                T_Wx_Head = T_Wx_Px[head_idx]                           # From XSens head coordinates to XSens world coordinates
                T_Hx_Wx = T_Wx_Head.inverse()                           # From XSens world coordinates to XSens head coordinates
                T_Wd_Wx = T_Wd_Hx @ T_Hx_Wx                             # From XSens world to unified world coordinates
                T_Wd_Pelvis = T_Wd_Wx @ T_Wx_Px[pelvis_idx]             # Xsens pelvis in unified world coordinates

                rotation_matrix = T_Wd_Pelvis.rotation().to_matrix()    # nparray (3, 3)
                translation = T_Wd_Pelvis.translation().reshape(3)      # nparray (3, )

                ########################################################
                ####  Extract all joint translations and orientations ######
                ########################################################
                T_Wd_Px = [T_Wd_Wx @ T_wx_px for T_wx_px in T_Wx_Px]  # Transform all joints to world coordinates
                joint_translations = np.array([T_wd_px.translation().reshape(3) for T_wd_px in T_Wd_Px])  # Shape (22, 3)
                joint_orientations = np.array([T_wd_px.rotation().to_matrix() for T_wd_px in T_Wd_Px])  # Shape (22, 3, 3)

                ########################################################
                ####  Extract foot contact information ######
                ########################################################
                contact_info = data_provider.body_dp.xsens_data[XSensConstants.k_foot_contacts][idx]  # Shape (4, ) - foot contact states

                ########################################################
                ####  Extract egoview RGB image ######
                ########################################################
                try:
                    rgb_result = data_provider.recording_head.get_rgb_image(timestamp_ns, time_domain=TimeDomain.TIME_CODE)
                    rgb_image_data = rgb_result[0]  # ImageData object
                    rgb_array = rgb_image_data.to_numpy_array()  # Shape (1408, 1408, 3)
                    rgb_chw = rgb_array.transpose(2, 0, 1)  # Convert to CHW format (3, 1408, 1408)
                    egoview_rgb = rgb_chw
                except Exception as e:
                    print(f"Warning: Failed to extract RGB for timestamp {timestamp_ns}: {e}")
                    egoview_rgb = None

                ########################################################

            # Only append data if we have valid egoview RGB (skip frames where RGB extraction failed)
            if egoview_rgb is not None:
                antego_data['timestamp_ns'].append(timestamp_ns)
                antego_data['root_orientation'].append(rotation_matrix)
                antego_data['root_translation'].append(translation)
                antego_data['joint_translation'].append(joint_translations)
                antego_data['joint_orientation'].append(joint_orientations)
                antego_data['contact_information'].append(contact_info)
                antego_data['egoview_RGB'].append(egoview_rgb)
            
        except Exception as e:
            print(f"Warning: Failed to extract data for timestamp {timestamp_ns}: {e}")
            continue
    
    return antego_data


# Example usage
if __name__ == "__main__":
    # Example usage
    sequence_path = Path("dataset_nymeria/nymeria_firstdownload/20231211_s1_seth_bowman_act4_9jyykj")
    
    if sequence_path.exists():
        # Method 1: Using NymeriaDataProvider (includes Aria world coordinate alignment)
        try:
            root_data = extract_antego_data(sequence_path, frame_rate=30.0)
            print(f"Extracted {len(root_data)} frames of root motion data")
            
            # Print first few frames
            for i, (timestamp, rotation, translation) in enumerate(root_data[:3]):
                print(f"Frame {i}:")
                print(f"  Timestamp: {timestamp} ns")
                print(f"  Rotation matrix shape: {rotation.shape}")
                print(f"  Translation: {translation}")
                print()
                
        except Exception as e:
            print(f"Method 1 failed: {e}")
            
    else:
        print("Please provide a valid sequence folder path")