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
        qvel_all  = np.vstack([np.zeros((1,8)), np.diff(qpos_all, axis=0)]) / dt[:,None]
        
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
      DataLoader(tr_ds,batch_train,shuffle=True,pin_memory=True),
      DataLoader(va_ds,batch_val,shuffle=False,pin_memory=True),
      stats
    )
