#!/usr/bin/env python3
"""
Extract root orientation and translation from Nymeria dataset sequences.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import h5py
import numpy as np
import pandas as pd
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.sensor_data import TimeDomain
from nymeria.data_provider import NymeriaDataProvider
from nymeria.body_motion_provider import BodyDataProvider
from nymeria.xsens_constants import XSensConstants

def extract_antego_data(
    sequence_folder: Path,      # ex: PosixPath('dataset_nymeria/nymeria_firstdownload/20231211_s1_seth_bowman_act4_9jyykj')
    frame_rate: float = 30.0,
    start_frame: int = 0,
    num_frames: int = -1,       # If -1, extract all frames in the sequence. Otherwise, extract num_frames frames.
    sample_rate: int = 1,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Extract root transforms, joint transforms, foot contacts, and egoview RGB image for each frame from a Nymeria sequence in Aria world coordinates.
    
    Args:
        sequence_folder: Path to the sequence folder containing the Nymeria data
        frame_rate: Desired sampling rate in fps (default: 30.0)
        start_frame: Start frame index (default: 0)
        num_frames: Number of frames to extract (default: -1 for all frames)
        sample_rate: Sample rate in frames (default: 1) - 1 = every frame, 2 = every other frame, etc.
            - Note: sample_rate will multiply the "universe time speed" of our sequence,
            - Ex: If we have 1000 frames at 30 fps and sample_rate == 2, 
            - The sequence will have 1000 frames and be at 30fps, but each frame will be sampled every 2 frames. 
            - Recording will look 2x faster but still have 1000 frames at 30fps.

    Returns:
        Dictionary containing the following keys:
        - timestamp_ns:             list of timestamps in nanoseconds
        - root_translation:         list of (3, ) nparray representing root position
        - root_orientation:         list of (3, 3) nparray representing root orientation
        - cpf_translation:          list of (3, ) nparray representing central pupil frame (CPF) position
        - cpf_orientation:          list of (3, 3) nparray representing central pupil frame (CPF) orientation
        - joint_translation:        list of (22, 3) nparray representing joint positions
        - joint_orientation:        list of (22, 3, 3) nparray representing joint orientations
        - contact_information:      list of (4, ) nparray representing foot contact states (4 contact points)
        - egoview_RGB:              list of (3, 1408, 1408) nparray representing egoview RGB image
        - pointcloud:               (N, 3) nparray representing global point cloud in world coordinates (static for entire sequence), N = 50,000 by default
        - motion_narration:         pandas DataFrame with [start_idx, end_idx, 'Describe my focus attention'] columns or None
                                    NOTE: start_idx and end_idx are the frame indices of the start and end of the narration.
        - activity_summarization:   pandas DataFrame with [start_idx, end_idx, 'Describe my activity'] columns or None
        - atomic_action:            pandas DataFrame with [start_idx, end_idx, 'Describe my atomic actions'] columns or None
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
    start_time_ns, end_time_ns = timespan_ns # Ex: (4366026000000, 5274244432000)
    
    # Calculate frame timestamps based on desired frame rate
    duration_ns = end_time_ns - start_time_ns   # 
    frame_interval_ns = int(1e9 / frame_rate)   # nanoseconds per frame
    
    frame_timestamps = [] # contains all timestamps_ns in the sequence at the desired frame rate
    current_time = start_time_ns
    while current_time <= end_time_ns:
        frame_timestamps.append(current_time)
        current_time += frame_interval_ns
    if num_frames != -1:
        frame_timestamps = frame_timestamps[start_frame:start_frame+num_frames*sample_rate:sample_rate]
    else:
        frame_timestamps = frame_timestamps[start_frame::sample_rate]
    
    # Extract global point cloud (static for entire sequence)
    pointcloud = None
    if data_provider.recording_head is not None and data_provider.recording_head.has_pointcloud:
        try:
            print("Extracting global point cloud...")
            pointcloud = data_provider.recording_head.get_pointcloud_cached()
            print(f"Extracted {len(pointcloud)} points")
        except Exception as e:
            print(f"Warning: Failed to extract point cloud: {e}")

    antego_data = {
        'timestamp_ns': [],
        'root_translation': [],
        'root_orientation': [],
        'cpf_translation': [],
        'cpf_orientation': [],
        'joint_translation': [],
        'joint_orientation': [],
        'contact_information': [],
        'egoview_RGB': [],
        'pointcloud': pointcloud,       # Static point cloud for entire sequence
        'motion_narration': None,       # DataFrame with [start_idx, end_idx, description] columns or None
        'activity_summarization': None, # DataFrame with [start_idx, end_idx, description] columns or None
        'atomic_action': None,          # DataFrame with [start_idx, end_idx, description] columns or None
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
                print(f"Warning: No XSens data found for timestamp {timestamp_ns}")
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

                ########################################################
                ####  Extract CPF (Central Pupil Frame) in world coordinates ######
                ########################################################
                # Get device calibration to get the Device-to-CPF transform
                device_calib = data_provider.recording_head.vrs_dp.get_device_calibration()
                T_Device_CPF = device_calib.get_transform_device_cpf()  # sophus.SE3 object - from device to CPF

                # Compute CPF pose in world coordinates: T_World_CPF = T_World_Device @ T_Device_CPF
                T_Wd_CPF = T_Wd_Hd @ T_Device_CPF
                cpf_rotation_matrix = T_Wd_CPF.rotation().to_matrix()    # nparray (3, 3)
                cpf_translation = T_Wd_CPF.translation().reshape(3)      # nparray (3, )

                ########################################################
                ####  Extract all joint translations and orientations ######
                ############################################################
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
                    egoview_rgb = np.rot90(rgb_chw, k=3, axes=(1, 2))  # Rotate 270 degrees clockwise on spatial dimensions (H, W)
                except Exception as e:
                    print(f"Warning: Failed to extract RGB for timestamp {timestamp_ns}: {e}")
                    egoview_rgb = None
                ########################################################
            else:
                print(f"Warning: No recording head data found for timestamp {timestamp_ns}")
                continue

            # Only append data if we have valid egoview RGB (skip frames where RGB extraction failed)
            if egoview_rgb is not None:
                antego_data['timestamp_ns'].append(timestamp_ns)
                antego_data['root_orientation'].append(rotation_matrix)
                antego_data['root_translation'].append(translation)
                antego_data['cpf_translation'].append(cpf_translation)
                antego_data['cpf_orientation'].append(cpf_rotation_matrix)
                antego_data['joint_translation'].append(joint_translations)
                antego_data['joint_orientation'].append(joint_orientations)
                antego_data['contact_information'].append(contact_info)
                antego_data['egoview_RGB'].append(egoview_rgb)
            
        except Exception as e:
            print(f"Warning: Failed to extract data for timestamp {timestamp_ns}: {e}")
            continue

    # Extract narration data if available
    # Need to build device_time array for timestamp conversion
    if len(antego_data['timestamp_ns']) > 0 and data_provider.recording_head is not None:
        try:
            # Convert frame timestamps to device time for narration alignment
            device_time = []
            for ts_ns in antego_data['timestamp_ns']:
                device_time_ns = data_provider.recording_head.vrs_dp.convert_from_timecode_to_device_time_ns(int(ts_ns))
                device_time.append(device_time_ns)
            device_time = np.array(device_time) # (num_frames, ) - nparray of device timestamps in nanoseconds

            # Process each narration type
            narration_configs = [
                ('motion_narration.csv', 'Describe my focus attention', 'motion_narration'),
                ('activity_summarization.csv', 'Describe my activity', 'activity_summarization'),
                ('atomic_action.csv', 'Describe my atomic actions', 'atomic_action')
            ]

            for csv_name, text_column, dict_key in narration_configs:
                csv_path = sequence_folder / 'narration' / csv_name
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path, delimiter=',', quotechar='"')
                        # Match start_time and end_time (in seconds) to indices of 'device_time' array
                        df['start_idx'] = df['start_time'].apply(lambda x: np.argmin(np.abs(device_time - x * 1e9)))
                        df['end_idx'] = df['end_time'].apply(lambda x: np.argmin(np.abs(device_time - x * 1e9)))
                        # Store only relevant columns
                        antego_data[dict_key] = df[['start_idx', 'end_idx', text_column]].copy()
                        print(f"Loaded {len(df)} {dict_key} entries")
                    except Exception as e:
                        print(f"Warning: Failed to load {csv_name}: {e}")
        except Exception as e:
            print(f"Warning: Failed to process narration data: {e}")

    return antego_data


def antegodict_to_hdf5(antego_data: Dict[str, Any], output_path: str) -> None:
    """
    Convert antego_data dictionary to HDF5 file with further processing and saves it to the specified `output_path`.

    Args:
        antego_data: Dictionary containing the extracted data from extract_antego_data()
        output_path: Path to the output HDF5 file
    Returns:
        None

    This function will examine the atomic_action DataFrame to form datapoints. Each atomic action is a datapoint!
    1. Each datapoint will be a dictionary with the following keys (each key is associated with a single atomic action):
        - start_idx:                int representing the start frame index of the atomic action
        - end_idx:                  int representing the end frame index of the atomic action
        - atomic_action:            string representing the atomic action
        - timestamp_ns:             list of timestamps in nanoseconds associated with the atomic action
        - root_translation:         list of (3, ) nparray representing root position
        - root_orientation:         list of (3, 3) nparray representing root orientation
        - cpf_translation:          list of (3, ) nparray representing central pupil frame (CPF) position
        - cpf_orientation:          list of (3, 3) nparray representing central pupil frame (CPF) orientation
        - joint_translation:        list of (22, 3) nparray representing joint positions
        - joint_orientation:        list of (22, 3, 3) nparray representing joint orientations
        - contact_information:      list of (4, ) nparray representing foot contact states (4 contact points)
        - egoview_RGB:              list of (3, 1408, 1408) nparray representing egoview RGB image
        - pointcloud:               (N, 3) nparray representing global point cloud in world coordinates (static for entire sequence), N = 50,000 by default
       We find what frame indices in the sequence correspond to the atomic action by examining the atomic_action DataFrame of the antego_data dictionary.
       The atomic action dataframe has two columns: 'start_idx' and 'end_idx'. We can use these to find the frame indices in the sequence.
    
    2. A final super dictionary has (k, v) = (datapoint_id, datapoint_dictionary):
        - datapoint_id is an integer representing the datapoint id (ex: the 1st atomic action has datapoint_id = 0, the 2nd atomic action has datapoint_id = 1, etc.)
        - datapoint_dictionary is a dictionary with the keys defined in step 1.
    
    3. The HDF5 file will have the following structure:
    
    <sequence_name>.h5
    ├── attributes: num_datapoints, pointcloud_shape
    ├── datapoint_0/
    │   ├── attributes: start_idx, end_idx, atomic_action, num_frames
    │   ├── timestamp_ns:           (N, ) array
    │   ├── root_translation:       (N, 3) array
    │   ├── root_orientation:       (N, 3, 3) array
    │   ├── cpf_translation:        (N, 3) array
    │   ├── cpf_orientation:        (N, 3, 3) array
    │   ├── joint_translation:      (N, 22, 3) array
    │   ├── joint_orientation:      (N, 22, 3, 3) array
    │   ├── contact_information:    (N, 4) array
    │   ├── egoview_RGB:            (N, 3, 1408, 1408) array
    │   └── pointcloud:             (M, 3) array (static for sequence)
    ├── datapoint_1/
    │   └── ... (same structure)
    """

    # Check if atomic_action DataFrame exists
    if antego_data.get('atomic_action') is None:
        raise ValueError("atomic_action DataFrame not found in antego_data")

    atomic_action_df = antego_data['atomic_action']
    num_datapoints = len(atomic_action_df)

    # Get the text column name (should be 'Describe my atomic actions')
    text_column = [col for col in atomic_action_df.columns if 'atomic' in col.lower() and col not in ['start_idx', 'end_idx']][0]

    print(f"Creating HDF5 file with {num_datapoints} datapoints from atomic actions...")

    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['num_datapoints'] = num_datapoints
        f.attrs['pointcloud_shape'] = str(antego_data['pointcloud'].shape) if antego_data.get('pointcloud') is not None else 'None'

        # Create a group for each datapoint
        for i in range(num_datapoints):
            start_idx = int(atomic_action_df.iloc[i]['start_idx'])
            end_idx = int(atomic_action_df.iloc[i]['end_idx'])
            atomic_action_text = str(atomic_action_df.iloc[i][text_column])

            # Create group for this datapoint and store metadata 
            datapoint_grp = f.create_group(f'datapoint_{i:05d}')
            datapoint_grp.attrs['start_idx'] = start_idx
            datapoint_grp.attrs['end_idx'] = end_idx
            datapoint_grp.attrs['atomic_action'] = atomic_action_text
            datapoint_grp.attrs['num_frames'] = end_idx - start_idx + 1

            # Extract and store data for frames in this atomic action
            # All lists are sliced from start_idx to end_idx+1 (inclusive)
            datapoint_grp.create_dataset('timestamp_ns',
                                        data=np.array(antego_data['timestamp_ns'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('root_translation',
                                        data=np.array(antego_data['root_translation'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('root_orientation',
                                        data=np.array(antego_data['root_orientation'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('cpf_translation',
                                        data=np.array(antego_data['cpf_translation'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('cpf_orientation',
                                        data=np.array(antego_data['cpf_orientation'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('joint_translation',
                                        data=np.array(antego_data['joint_translation'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('joint_orientation',
                                        data=np.array(antego_data['joint_orientation'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('contact_information',
                                        data=np.array(antego_data['contact_information'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            datapoint_grp.create_dataset('egoview_RGB',
                                        data=np.array(antego_data['egoview_RGB'][start_idx:end_idx+1]),
                                        compression='gzip', compression_opts=4)

            # Store pointcloud once per datapoint (it's static for entire sequence, but include it for convenience)
            if antego_data.get('pointcloud') is not None:
                datapoint_grp.create_dataset('pointcloud',
                                            data=antego_data['pointcloud'],
                                            compression='gzip', compression_opts=4)

            if i % 10 == 0 or i == num_datapoints - 1:
                print(f"  Processed datapoint {i+1}/{num_datapoints} (frames {start_idx}-{end_idx}, action: '{atomic_action_text[:50]}...')")

    print(f"✓ Saved HDF5 file: {output_path} ({os.path.getsize(output_path) / 1e6:.2f} MB)")


def extract_to_hdf5_chunked(sequence_folder: Path, output_dir: str, frame_rate: float = 30.0) -> None:
    """This function is a more memory-efficient version of extract_antego_data() and antegodict_to_hdf5() combined.
    Because it does not load the entire sequence (such as all video frames) like extract_antego_data() does, it is memory friendly.

    Args:
        sequence_folder:    Path to the nymeria sequence folder
        output_dir:         Path to the output directory where the hdf5 files will be saved
        frame_rate:         Frame rate for extraction (fps)
    Returns:
        A list of hdf5 file paths in sorted alphabetically order.

    When processing a nymeria sequence folder, it has many datapoints. Each datapoint is determined by an "atomic action" language description.
    We process the nymeria sequence by doing the following:
    
    1. We look at the atomic_action.csv file to get a dataframe of all the atomic actions in the sequence.
        - In doing so, we also get the start_idx and end_idx of each atomic action.
    2. We extract the data for each atomic action by iterating through the frame timestamps and extracting the data for each frame.
    3. We save the data for each atomic action to a separate hdf5 file.
    4. We return a list of the hdf5 file paths in sorted alphabetically order.
    
    Each hdf5 file is a datapoint with the following structure:
    <sequence_name>_<datapoint_id>.h5
    ├── attributes: sequence_name, start_idx, end_idx, atomic_action, num_frames
    ├── timestamp_ns:           (N, ) array
    ├── root_translation:       (N, 3) array
    ├── root_orientation:       (N, 3, 3) array
    ├── cpf_translation:        (N, 3) array
    ├── cpf_orientation:        (N, 3, 3) array
    ├── joint_translation:      (N, 22, 3) array
    ├── joint_orientation:      (N, 22, 3, 3) array
    ├── contact_information:    (N, 4) array
    └── egoview_RGB:            (N, 3, 1408, 1408) array
    """
    sequence_name = sequence_folder.name
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
    start_time_ns, end_time_ns = timespan_ns # Ex: (4366026000000, 5274244432000)
    
    # Calculate frame timestamps based on desired frame rate
    duration_ns = end_time_ns - start_time_ns   # 
    frame_interval_ns = int(1e9 / frame_rate)   # nanoseconds per frame
    
    frame_timestamps = [] # contains all timestamps_ns in the sequence at the desired frame rate
    current_time = start_time_ns
    while current_time <= end_time_ns:
        frame_timestamps.append(current_time)
        current_time += frame_interval_ns
    
    ########################################################
    ####  Step 1: Get the atomic actions ######
    ########################################################
    # Extract narration data if available
    if len(frame_timestamps) > 0 and data_provider.recording_head is not None:
        try:
            # Convert frame timestamps to device time for narration alignment
            device_time = []
            for ts_ns in frame_timestamps:
                device_time_ns = data_provider.recording_head.vrs_dp.convert_from_timecode_to_device_time_ns(int(ts_ns))
                device_time.append(device_time_ns)
            device_time = np.array(device_time) # (num_frames, ) - nparray of device timestamps in nanoseconds

            # Read the atomic action dataframe and calculate the start_idx and end_idx for each atomic action
            atomic_action_df = pd.read_csv(sequence_folder / 'narration' / 'atomic_action.csv', delimiter=',', quotechar='"')
            atomic_action_df['start_idx'] = atomic_action_df['start_time'].apply(lambda x: np.argmin(np.abs(device_time - x * 1e9)))
            atomic_action_df['end_idx'] = atomic_action_df['end_time'].apply(lambda x: np.argmin(np.abs(device_time - x * 1e9)))

        except Exception as e:
            print(f"Warning: Failed to process narration data: {e}")
        
    ########################################################
    ####  Step 2: Process each atomic action as a datapoint ######
    ########################################################
    hdf5_file_paths = []
    for i in range(len(atomic_action_df)):
        start_idx = int(atomic_action_df.iloc[i]['start_idx'])
        end_idx = int(atomic_action_df.iloc[i]['end_idx'])
        atomic_action_text = str(atomic_action_df.iloc[i]["Describe my atomic actions"])

        # Create a new hdf5 file for this atomic action
        hdf5_path = Path(output_dir) / f"{sequence_name}_{i:05d}.h5"
        with h5py.File(hdf5_path, 'w') as f:
            # Store metadata
            f.attrs['sequence_name'] = sequence_name
            f.attrs['start_idx'] = start_idx
            f.attrs['end_idx'] = end_idx
            f.attrs['atomic_action'] = atomic_action_text
            f.attrs['num_frames'] = end_idx - start_idx + 1
            
            # Extract the data for this atomic action
            timestamp_ns_list = []
            root_translation_list = []
            root_orientation_list = []
            cpf_translation_list = []
            cpf_orientation_list = []
            joint_translation_list = []
            joint_orientation_list = []
            contact_information_list = []
            egoview_RGB_list = []
            for timestamp_ns in frame_timestamps[start_idx:end_idx+1]:
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
                        print(f"Warning: No XSens data found for timestamp {timestamp_ns}")
                        continue
                    # The xsens skeleton data is in format (22, 2, 3) representing bones as
                    # [child_position, parent_position] pairs. To get proper root orientation
                    # and translation, we need to access the raw SE3 transforms directly.

                    timestamp_us = int(timestamp_ns / 1e3)

                    if data_provider.recording_head is not None and 'recording_head' in poses: # If no head pose available, skip this frame
                        #------ Extract root transform in world coordinates ------#
                        # Get head pose in world coordinates
                        head_pose = poses['recording_head']             # head_pose is a mps.ClosedLoopTrajectoryPose object
                        T_Wd_Hd = head_pose.transform_world_device      # sophus.SE3 object - from device to world coordinates, see https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory

                        #------ Extract CPF (Central Pupil Frame) in world coordinates ------#
                        # Get device calibration to get the Device-to-CPF transform
                        device_calib = data_provider.recording_head.vrs_dp.get_device_calibration()
                        T_Device_CPF = device_calib.get_transform_device_cpf()  # sophus.SE3 object - from device to CPF

                        # Compute CPF pose in world coordinates: T_World_CPF = T_World_Device @ T_Device_CPF
                        T_Wd_CPF = T_Wd_Hd @ T_Device_CPF
                        cpf_rotation_matrix = T_Wd_CPF.rotation().to_matrix()    # nparray (3, 3)
                        cpf_translation = T_Wd_CPF.translation().reshape(3)      # nparray (3, )

                        #----  Extract all joint translations and orientations ----#
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

                        T_Wd_Px = [T_Wd_Wx @ T_wx_px for T_wx_px in T_Wx_Px]  # Transform all joints to world coordinates
                        joint_translations = np.array([T_wd_px.translation().reshape(3) for T_wd_px in T_Wd_Px])  # Shape (22, 3)
                        joint_orientations = np.array([T_wd_px.rotation().to_matrix() for T_wd_px in T_Wd_Px])  # Shape (22, 3, 3)

                        #------ Extract foot contact information ------#
                        contact_info = data_provider.body_dp.xsens_data[XSensConstants.k_foot_contacts][idx]  # Shape (4, ) - foot contact states

                        #------ Extract egoview RGB image ------#
                        try:
                            rgb_result = data_provider.recording_head.get_rgb_image(timestamp_ns, time_domain=TimeDomain.TIME_CODE)
                            rgb_image_data = rgb_result[0]  # ImageData object
                            rgb_array = rgb_image_data.to_numpy_array()  # Shape (1408, 1408, 3)
                            rgb_chw = rgb_array.transpose(2, 0, 1)  # Convert to CHW format (3, 1408, 1408)
                            egoview_rgb = np.rot90(rgb_chw, k=3, axes=(1, 2))  # Rotate 270 degrees clockwise on spatial dimensions (H, W)
                        except Exception as e:
                            print(f"Warning: Failed to extract RGB for timestamp {timestamp_ns}: {e}")
                            egoview_rgb = None
                        ########################################################
                    else:
                        print(f"Warning: No recording head data found for timestamp {timestamp_ns}")
                        continue

                    # Only append data if we have valid egoview RGB (skip frames where RGB extraction failed)
                    if egoview_rgb is not None:
                        timestamp_ns_list.append(timestamp_ns)
                        root_translation_list.append(translation)
                        root_orientation_list.append(rotation_matrix)
                        cpf_translation_list.append(cpf_translation)
                        cpf_orientation_list.append(cpf_rotation_matrix)
                        joint_translation_list.append(joint_translations)
                        joint_orientation_list.append(joint_orientations)
                        contact_information_list.append(contact_info)
                        egoview_RGB_list.append(egoview_rgb)
                    
                except Exception as e:
                    print(f"Warning: Failed to extract data for timestamp {timestamp_ns}: {e}")
                    continue
                
            ########################################################
            ####  Step 3: Save the datapoint as an hdf5 file ######
            ########################################################
            # Save the data to the hdf5 file
            f.create_dataset('timestamp_ns',data=np.array(timestamp_ns_list), compression='gzip', compression_opts=4)
            f.create_dataset('root_translation', data=np.array(root_translation_list), compression='gzip', compression_opts=4)
            f.create_dataset('root_orientation', data=np.array(root_orientation_list), compression='gzip', compression_opts=4)
            f.create_dataset('cpf_translation', data=np.array(cpf_translation_list), compression='gzip', compression_opts=4)
            f.create_dataset('cpf_orientation', data=np.array(cpf_orientation_list), compression='gzip', compression_opts=4)
            f.create_dataset('joint_translation', data=np.array(joint_translation_list),compression='gzip', compression_opts=4)
            f.create_dataset('joint_orientation', data=np.array(joint_orientation_list),compression='gzip', compression_opts=4)
            f.create_dataset('contact_information', data=np.array(contact_information_list),compression='gzip', compression_opts=4)
            f.create_dataset('egoview_RGB', data=np.array(egoview_RGB_list),compression='gzip', compression_opts=4)
            print(f"Saved HDF5 file: {hdf5_path} ({os.path.getsize(hdf5_path) / 1e6:.2f} MB)")
            hdf5_file_paths.append(hdf5_path)
    hdf5_file_paths.sort()
    return hdf5_file_paths

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