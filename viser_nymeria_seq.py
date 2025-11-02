import viser
import tyro
import viser.transforms as vtf
import numpy as np
import time
from pathlib import Path
from extract_antego_data import extract_antego_data
from nymeria.xsens_constants import XSensConstants
import torch
import torch.nn.functional as F

# from pi3.models.pi3 import Pi3
# from pi3.utils.geometry import depth_edge

def main(
    sequence_path: str,
    start_frame: int = 0,
    num_frames: int = 1000,
    sample_rate: int = 20,
):
    # First set up the server and configure the theme + reset camera button
    server = viser.ViserServer(port=8080)
    server.gui.configure_theme(
        control_layout="collapsible",
        show_logo=False
    )
    reset_camera = server.gui.add_button(
            label="Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )
    @reset_camera.on_click
    def _reset_camera_cb(_) -> None:
        for client in server.get_clients().values():
            client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    # Now load the sequence to get the number of frames
    print(f"Loading Nymeria sequence data starting from frame {start_frame} for a total of {num_frames} frames...")
    nymeria_dict = extract_antego_data(Path(sequence_path), start_frame=start_frame, num_frames=num_frames, sample_rate=sample_rate)
    total_num_frames = len(nymeria_dict['timestamp_ns'])

    # # load our pi3 model
    # print("Loading model...")
    # ckpt = "/home/anthony_zhang/Pi3/model.safetensors"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if ckpt is not None:
    #     model = Pi3().to(device).eval()
    #     if ckpt.endswith('.safetensors'):
    #         from safetensors.torch import load_file
    #         weight = load_file(ckpt)
    #     else:
    #         weight = torch.load(ckpt, map_location=device, weights_only=False)
    #     model.load_state_dict(weight)

    # # load our video
    # breakpoint()
    # # Convert to torch tensors and downsample using interpolation
    # downsampled_images = []
    # for img in nymeria_dict['egoview_RGB']:
    #     img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add batch dim: (1, 3, 1408, 1408)
    #     downsampled = F.interpolate(img_tensor, size=(352, 352), mode='bilinear', align_corners=False)
    #     downsampled_images.append(downsampled.squeeze(0))  # Remove batch dim: (3, 352, 352)
    # # Stack into final tensor
    # imgs = torch.stack(downsampled_images, dim=0)  # Shape: (N, 3, 352, 352)
    
    # Add playback UI
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=30
        ) # NOTE: this is the frame rate of the visualization which as subsampled from the original video sequence

    # Add visualization controls
    with server.gui.add_folder("Visualization"):
        gui_show_joints = server.gui.add_checkbox("Show Joints", True)
        gui_show_bones = server.gui.add_checkbox("Show Bones", True)
        gui_show_root_frame = server.gui.add_checkbox("Show Root Frame", True)
        gui_show_cpf_frame = server.gui.add_checkbox("Show CPF Frame", True)
        gui_show_contacts = server.gui.add_checkbox("Show Foot Contacts", True)
        gui_show_pointcloud = server.gui.add_checkbox("Show Point Cloud", True)
    
    # Create static point cloud once (it doesn't change across frames)
    pointcloud_handle = None
    if nymeria_dict['pointcloud'] is not None:
        pointcloud_handle = server.scene.add_point_cloud(
            "environment/pointcloud",
            points=nymeria_dict['pointcloud'].astype(np.float32),
            colors=(0.7, 0.7, 0.7),  # Gray color for environment
            point_size=0.01,
            point_shape="circle"
        )
    
    # Store handles for dynamic scene elements
    dynamic_handles = {
        'joints': None,
        'bones': None,
        'root_frame': None,
        'cpf_frame': None,
        'contacts': None
    }
    
    def update_frame_visualization(timestep: int) -> None:
        """Update the 3D visualization for the current timestep."""
        if timestep >= num_frames:
            return

        # Get data for current frame
        joint_translations = nymeria_dict['joint_translation'][timestep]  # Shape (22, 3)
        joint_orientations = nymeria_dict['joint_orientation'][timestep]  # Shape (22, 3, 3)
        root_translation = nymeria_dict['root_translation'][timestep]     # Shape (3,)
        root_orientation = nymeria_dict['root_orientation'][timestep]     # Shape (3, 3)
        cpf_translation = nymeria_dict['cpf_translation'][timestep]       # Shape (3,)
        cpf_orientation = nymeria_dict['cpf_orientation'][timestep]       # Shape (3, 3)
        contact_info = nymeria_dict['contact_information'][timestep]      # Shape (4,)

        # Clear previous frame's dynamic scene nodes (but keep the static point cloud)
        for handle in dynamic_handles.values():
            if handle is not None:
                handle.remove()

        # Render skeleton joints as point cloud
        if gui_show_joints.value:
            # Prepare joint colors - blue for pelvis, green for others
            joint_colors = np.zeros((len(joint_translations), 3))
            for i, joint_name in enumerate(XSensConstants.part_names):
                if joint_name == "Pelvis":
                    joint_colors[i] = [0.2, 0.2, 0.8]  # Blue for pelvis
                else:
                    joint_colors[i] = [0.2, 0.8, 0.2]  # Green for other joints

            dynamic_handles['joints'] = server.scene.add_point_cloud(
                "joints/point_cloud",
                points=joint_translations.astype(np.float32),
                colors=joint_colors.astype(np.float32),
                point_size=0.04,
                point_shape="circle"
            )

        # Render skeleton bones as lines using kinematic tree
        if gui_show_bones.value:
            bone_points = []
            for i, parent_idx in enumerate(XSensConstants.kintree_parents):
                if parent_idx >= 0:  # Skip root joint
                    child_pos = joint_translations[i]
                    parent_pos = joint_translations[parent_idx]
                    bone_points.append([parent_pos, child_pos])

            if bone_points:
                bone_points = np.array(bone_points) # (N=22, 2, 3)
                dynamic_handles['bones'] = server.scene.add_line_segments(
                    "skeleton/bones",
                    points=bone_points.astype(float),
                    colors=(0.1, 0.1, 0.8),
                    line_width=3.0,
                )

        # Render root coordinate frame
        if gui_show_root_frame.value:
            dynamic_handles['root_frame'] = server.scene.add_frame(
                "root/frame",
                wxyz=vtf.SO3.from_matrix(root_orientation).wxyz,
                position=root_translation.astype(float),
                axes_length=0.1,
                axes_radius=0.005,
            )

        # Render CPF (Central Pupil Frame) coordinate frame
        if gui_show_cpf_frame.value:
            dynamic_handles['cpf_frame'] = server.scene.add_frame(
                "cpf/frame",
                wxyz=vtf.SO3.from_matrix(cpf_orientation).wxyz,
                position=cpf_translation.astype(float),
                axes_length=0.15,
                axes_radius=0.007,
            )

        # Render foot contacts as point cloud
        if gui_show_contacts.value:
            # Map contacts to foot joint positions
            foot_joints = ["R_Foot", "R_Toe", "L_Foot", "L_Toe"]  # 4 contact points
            contact_points = []
            contact_colors = []

            for i, (contact_state, foot_name) in enumerate(zip(contact_info, foot_joints)):
                if foot_name in XSensConstants.part_names:
                    foot_idx = XSensConstants.part_names.index(foot_name)
                    foot_pos = joint_translations[foot_idx]
                    contact_points.append(foot_pos)

                    # Red for contact, blue for no contact
                    color = [0.8, 0.1, 0.1] if contact_state > 0.5 else [0.1, 0.1, 0.8]
                    contact_colors.append(color)

            if contact_points:
                dynamic_handles['contacts'] = server.scene.add_point_cloud(
                    "contacts/point_cloud",
                    points=np.array(contact_points).astype(np.float32),
                    colors=np.array(contact_colors).astype(np.float32),
                    point_size=0.06,
                    point_shape="diamond"
                )

        # Toggle point cloud visibility (it was created once at the beginning)
        if pointcloud_handle is not None:
            pointcloud_handle.visible = gui_show_pointcloud.value
        # print(f"Updated visualization for frame {timestep} (timestamp: {nymeria_dict['timestamp_ns'][timestep]})")

    # Update visualization when timestep changes
    @gui_timestep.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        update_frame_visualization(current_timestep)

    # Button callbacks
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = min(gui_timestep.value + 1, num_frames - 1)

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = max(gui_timestep.value - 1, 0)

    # Visualization control callbacks
    @gui_show_joints.on_update
    @gui_show_bones.on_update
    @gui_show_root_frame.on_update
    @gui_show_cpf_frame.on_update
    @gui_show_contacts.on_update
    @gui_show_pointcloud.on_update
    def _(_) -> None:
        update_frame_visualization(gui_timestep.value)

    # Initialize with first frame
    print(f"Loaded {num_frames} frames. Starting visualization...")
    update_frame_visualization(0)

    # Animation loop
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    tyro.cli(main)