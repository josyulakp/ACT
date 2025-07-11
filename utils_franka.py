# utils_franka.py
import os, numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader

class FrankaDataset(Dataset):
    def __init__(self, episode_dirs, camera_names, norm_stats, sample_full_episode=False):
        """
        episode_dirs: list of full paths to episode_001, episode_002, …
        camera_names:   ['left_camera','right_camera','wrist_camera']
        norm_stats:    dict with 'qpos_mean','qpos_std','action_mean','action_std'
        """
        self.episode_dirs = episode_dirs
        self.camera_names  = camera_names
        self.norm_stats   = norm_stats
        self.sample_full_episode = sample_full_episode

    def __len__(self):
        return len(self.episode_dirs)

    def __getitem__(self, idx):
        ep_dir = self.episode_dirs[idx]
        # --- load proprio & commands ---
        data = np.load(os.path.join(ep_dir, 'episode_data.npz'))
        # joint_states ∈ ℝ^(T,7)   ← robot qpos
        qpos_all  = data['joint_states'].astype(np.float32)
        
        # Check if gripper data exists in the dataset
        if 'gripper_states' in data:
            gripper_states = data['gripper_states'].astype(np.float32).reshape(-1, 1)
        else:
            # If no gripper data, add a placeholder column of zeros
            gripper_states = np.zeros((qpos_all.shape[0], 1), dtype=np.float32)
        
        # Append gripper states to make it 8-dimensional
        qpos_all = np.hstack([qpos_all, gripper_states])
        
        # approximate qvel by finite diff
        ts        = data['timestamps']
        dt        = np.diff(ts, prepend=ts[0])
        qvel_all  = np.vstack([np.zeros((1,8)), np.diff(qpos_all, axis=0)])
        
        # action ∈ ℝ^(T,7) ← commanded via gello_joint_states
        action_all = data['gello_joint_states'].astype(np.float32)
        
        # Check if gripper commands exist in the dataset
        if 'gello_gripper_percent' in data:
            gripper_commands = data['gello_gripper_percent'].astype(np.float32).reshape(-1, 1)
        else:
            # If no gripper commands, add a placeholder column of zeros
            gripper_commands = np.zeros((action_all.shape[0], 1), dtype=np.float32)
        
        # Append gripper commands to make it 8-dimensional
        action_all = np.hstack([action_all, gripper_commands])
        T = qpos_all.shape[0]

        # --- pick a random start ---
        if self.sample_full_episode:
            t0 = 0
        else:
            t0 = np.random.randint(0, T)

        # slice
        qpos   = qpos_all[t0]
        qvel   = qvel_all[t0]
        acts   = action_all[t0:]
        L      = acts.shape[0]

        # pad actions & mask
        padded = np.zeros_like(action_all)
        padded[:L] = acts
        is_pad = np.zeros(T, dtype=bool)
        is_pad[L:] = True

        # --- load images at t0 from each cam ---
        imgs = []
        for cam in self.camera_names:
            cap = cv2.VideoCapture(os.path.join(ep_dir, f"{cam}.avi"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise IOError(f"Frame {t0} missing in {cam}.avi")
            # BGR→RGB, HWC→CHW, normalize
            frame = frame[..., ::-1] / 255.0
            imgs.append(frame.transpose(2,0,1))
        image_data = torch.from_numpy(np.stack(imgs,0)).float()

        # --- normalize & to-tensor ---
        qpos_n = (qpos - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
        act_n  = (padded - self.norm_stats['action_mean']) / self.norm_stats['action_std']

        return (
            image_data,                              # [cams,3,H,W]
            torch.from_numpy(qpos_n).float(),        # [8] (now includes gripper)
            torch.from_numpy(act_n).float(),         # [T,8] (now includes gripper)
            torch.from_numpy(is_pad)                 # [T]
        )

def get_norm_stats_franka(episode_dirs):
    """Compute mean/std of qpos & action over all episodes."""
    all_q = []; all_a = []
    for d in episode_dirs:
        data = np.load(os.path.join(d, 'episode_data.npz'))
        
        # Load joint states (qpos)
        qpos = data['joint_states']
        
        # Check if gripper data exists
        if 'gripper_states' in data:
            gripper_states = data['gripper_states'].reshape(-1, 1)
        else:
            gripper_states = np.zeros((qpos.shape[0], 1))
        
        # Combine joint states with gripper states
        qpos_with_gripper = np.hstack([qpos, gripper_states])
        all_q.append(qpos_with_gripper)
        
        # Load actions
        actions = data['gello_joint_states']
        
        # Check if gripper commands exist
        if 'gello_gripper_percent' in data:
            gripper_commands = data['gello_gripper_percent'].reshape(-1, 1)
        else:
            gripper_commands = np.zeros((actions.shape[0], 1))
        
        # Combine actions with gripper commands
        actions_with_gripper = np.hstack([actions, gripper_commands])
        all_a.append(actions_with_gripper)
        
    all_q = np.vstack(all_q); all_a = np.vstack(all_a)
    qm, qs = all_q.mean(0), all_q.std(0).clip(1e-2)
    am, as_ = all_a.mean(0), all_a.std(0).clip(1e-2)
    return {
      'qpos_mean': qm, 'qpos_std': qs,
      'action_mean': am, 'action_std': as_
    }

def make_franka_loaders(root_dir, batch_train, batch_val, cameras):
    # scan episodes
    eps = sorted([os.path.join(root_dir,d) for d in os.listdir(root_dir)
                  if d.startswith('episode_')])
    split = int(0.8*len(eps))
    tr, va = eps[:split], eps[split:]
    stats = get_norm_stats_franka(eps)
    tr_ds = FrankaDataset(tr, cameras, stats)
    va_ds = FrankaDataset(va, cameras, stats)
    return (
      DataLoader(tr_ds,batch_train,shuffle=True,pin_memory=True,collate_fn=franka_collate_fn),
      DataLoader(va_ds,batch_val,shuffle=False,pin_memory=True,collate_fn=franka_collate_fn),
      stats
    )

def get_data_params(dataset_dir):
    '''Get data parameters for the Franka dataset.
    Args:
        dataset_dir (str): Path to the dataset directory.
    Returns:
        cameras (list): List of camera names found in the dataset.
        num_episodes (int): Number of episodes in the dataset.
        episode_length (int): Length of each episode in frames.'''
    cameras = []
    # Check if the dataset directory exists and there is at least one episode
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist.")
    episode_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('episode_')]
    if len(episode_dirs) == 0:
        raise ValueError(f"No episode_ folders found in {dataset_dir}.")
    else:
        print(f"Found {len(episode_dirs)} episodes in {dataset_dir}.")
    first_episode_dir = os.path.join(dataset_dir, 'episode_001')
    for f in os.listdir(first_episode_dir):
        # end with .avi and not episode_*
        if f.endswith('.avi') and not f.startswith('episode_'):
            cameras.append(f[:-4])
    cameras.sort()
    num_episodes = len([d for d in os.listdir(dataset_dir) if d.startswith('episode_')])
    episode_length = 0
    if num_episodes > 0:
        first_episode = os.path.join(dataset_dir, 'episode_001')
        if os.path.exists(first_episode):
            data = np.load(os.path.join(first_episode, 'episode_data.npz'))
            episode_length = data['joint_states'].shape[0]
    return cameras, num_episodes, episode_length

from torch.nn.functional import pad

def franka_collate_fn(batch):
    images, qpos, actions, masks = zip(*batch)  # list of tensors

    # Images are same shape already
    images = torch.stack(images, dim=0)
    qpos   = torch.stack(qpos, dim=0)

    # Pad action sequences to max length
    max_len = max(a.shape[0] for a in actions)
    padded_actions = []
    padded_masks   = []
    for a, m in zip(actions, masks):
        pad_amt = max_len - a.shape[0]
        padded_actions.append(pad(a, (0, 0, 0, pad_amt)))  # pad on dim=0
        padded_masks.append(pad(m, (0, pad_amt), value=True))  # pad mask as True (meaning "ignore")

    action_tensor = torch.stack(padded_actions, dim=0)
    mask_tensor   = torch.stack(padded_masks, dim=0)

    return images, qpos, action_tensor, mask_tensor
