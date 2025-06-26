#!/usr/bin/env python3
"""
Sanity check evaluation script for Franka robot.
Tests the trained model on training data to verify it can predict actions correctly.
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import glob

from utils_franka import make_franka_loaders, get_data_params, FrankaDataset
from utils import set_seed, compute_dict_mean
from policy import ACTPolicy, CNNMLPPolicy


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def load_model_and_stats(ckpt_dir, policy_class):
    """Load the trained model and dataset statistics"""
    # Load dataset stats
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Load model checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    if not os.path.exists(ckpt_path):
        # Find all policy_epoch_* files and pick the one with highest epoch
        epoch_ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('policy_epoch_') and f.endswith('.ckpt')]
        if epoch_ckpts:
            # Extract epoch numbers and pick the highest
            def get_epoch_num(fname):
                try:
                    return int(fname.split('policy_epoch_')[1].split('.ckpt')[0])
                except Exception:
                    return -1
            epoch_ckpts.sort(key=get_epoch_num, reverse=True)
            ckpt_path = os.path.join(ckpt_dir, epoch_ckpts[0])
    ckpt_path = "/home/jeeveshm/franka_teleop/ACT/outputs/policy_epoch_40000_seed_0.ckpt"
    print(f"Loading model from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    return stats, ckpt_path


def evaluate_on_episode(policy, episode_dir, camera_names, stats, device='cuda'):
    """Evaluate model on a single episode"""
    
    # Load episode data
    data = np.load(os.path.join(episode_dir, 'episode_data.npz'))
    
    # Load joint states and gripper states
    qpos_all = data['joint_states'].astype(np.float32)
    if 'gripper_states' in data:
        gripper_states = data['gripper_states'].astype(np.float32).reshape(-1, 1)
    else:
        gripper_states = np.zeros((qpos_all.shape[0], 1), dtype=np.float32)
    qpos_all = np.hstack([qpos_all, gripper_states])
    
    # Load actions and gripper commands
    action_all = data['gello_joint_states'].astype(np.float32)
    if 'gello_gripper_percent' in data:
        gripper_commands = data['gello_gripper_percent'].astype(np.float32).reshape(-1, 1)
    else:
        gripper_commands = np.zeros((action_all.shape[0], 1), dtype=np.float32)
    action_all = np.hstack([action_all, gripper_commands])
    
    T = qpos_all.shape[0]
    
    # Preprocessing functions
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    predictions = []
    ground_truths = []
    
    # Load all video captures once
    video_caps = {}
    for cam in camera_names:
        video_path = os.path.join(episode_dir, f"{cam}.avi")
        if os.path.exists(video_path):
            video_caps[cam] = cv2.VideoCapture(video_path)
        else:
            print(f"Warning: Video file {video_path} not found")
    
    try:
        with torch.no_grad():
            for t in range(T):
                # Get current qpos
                qpos_raw = qpos_all[t]
                qpos_normalized = pre_process(qpos_raw)
                qpos_tensor = torch.from_numpy(qpos_normalized).float().to(device).unsqueeze(0)
                
                # Get current images
                images = []
                for cam in camera_names:
                    if cam in video_caps:
                        cap = video_caps[cam]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                        ret, frame = cap.read()
                        if ret:
                            # BGR→RGB, HWC→CHW, normalize
                            frame = frame[..., ::-1] / 255.0
                            images.append(frame.transpose(2, 0, 1))
                        else:
                            print(f"Warning: Could not read frame {t} from {cam}")
                            # Use zeros as fallback
                            images.append(np.zeros((3, 480, 640), dtype=np.float32))
                    else:
                        # Use zeros if video not available
                        images.append(np.zeros((3, 480, 640), dtype=np.float32))
                
                if len(images) == len(camera_names):
                    image_tensor = torch.from_numpy(np.stack(images, 0)).float().to(device).unsqueeze(0)
                    
                    # Get model prediction
                    if hasattr(policy, 'num_queries') or 'ACT' in str(type(policy)):
                        # For ACT policy, we predict a sequence but take the first action
                        pred_actions = policy(qpos_tensor, image_tensor)
                        pred_action = pred_actions[0, 0].cpu().numpy()  # Take first timestep
                    else:
                        # For CNNMLP policy
                        pred_action = policy(qpos_tensor, image_tensor)
                        pred_action = pred_action[0].cpu().numpy()
                    
                    # Post-process prediction
                    pred_action_denorm = post_process(pred_action)
                    
                    # Get ground truth action
                    if t < T - 1:  # We don't have ground truth for the last timestep
                        gt_action = action_all[t]
                        predictions.append(pred_action_denorm)
                        ground_truths.append(gt_action)
    
    finally:
        # Clean up video captures
        for cap in video_caps.values():
            cap.release()
    
    return np.array(predictions), np.array(ground_truths)


def compute_metrics(predictions, ground_truths):
    """Compute evaluation metrics"""
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {}
    
    # Mean Squared Error
    mse = np.mean((predictions - ground_truths) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - ground_truths))
    
    # Per-joint metrics
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'gripper']
    per_joint_mse = np.mean((predictions - ground_truths) ** 2, axis=0)
    per_joint_mae = np.mean(np.abs(predictions - ground_truths), axis=0)
    
    metrics = {
        'overall_mse': mse,
        'overall_mae': mae,
        'per_joint_mse': dict(zip(joint_names, per_joint_mse)),
        'per_joint_mae': dict(zip(joint_names, per_joint_mae))
    }
    
    return metrics


def plot_predictions(predictions, ground_truths, save_path, episode_name):
    """Plot predictions vs ground truth for visualization, including filtered predictions."""
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'Gripper']

    # Simple low-pass filter (exponential moving average)
    def low_pass_filter(data, alpha=0.2):
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for t in range(1, len(data)):
            filtered[t] = alpha * data[t] + (1 - alpha) * filtered[t - 1]
        return filtered

    filtered_predictions = low_pass_filter(predictions)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        ax = axes[i]
        timesteps = range(len(predictions))

        ax.plot(timesteps, ground_truths[:, i], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(timesteps, predictions[:, i], 'r--', label='Prediction', alpha=0.7)
        ax.plot(timesteps, filtered_predictions[:, i], 'g-', label='Filtered', alpha=0.7)

        ax.set_title(f'{joint_names[i]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Joint Value')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'Predictions vs Ground Truth - {episode_name}', y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_epoch_videos(episode_dirs, camera_names, stats, args, joint_names=None):
    import cv2

    if joint_names is None:
        joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'Gripper']

    # Find all epoch checkpoints
    ckpt_pattern = os.path.join(args.ckpt_dir, "policy_epoch_*_seed_*.ckpt")
    ckpt_files = sorted(glob.glob(ckpt_pattern), key=lambda x: int(x.split("policy_epoch_")[1].split("_")[0]))
    epoch_numbers = [int(f.split("policy_epoch_")[1].split("_")[0]) for f in ckpt_files]

    for episode_dir in episode_dirs:
        episode_name = os.path.basename(episode_dir)
        print(f"Creating video for {episode_name} over epochs...")
        # Load GT once
        data = np.load(os.path.join(episode_dir, 'episode_data.npz'))
        qpos_all = data['joint_states'].astype(np.float32)
        if 'gripper_states' in data:
            gripper_states = data['gripper_states'].astype(np.float32).reshape(-1, 1)
        else:
            gripper_states = np.zeros((qpos_all.shape[0], 1), dtype=np.float32)
        qpos_all = np.hstack([qpos_all, gripper_states])
        action_all = data['gello_joint_states'].astype(np.float32)
        if 'gello_gripper_percent' in data:
            gripper_commands = data['gello_gripper_percent'].astype(np.float32).reshape(-1, 1)
        else:
            gripper_commands = np.zeros((action_all.shape[0], 1), dtype=np.float32)
        action_all = np.hstack([action_all, gripper_commands])
        T = qpos_all.shape[0]
        gt = action_all[:T-1]  # ground truth actions

        # For each epoch, get predictions
        preds_over_epochs = []
        for ckpt_path in ckpt_files:
            # Load policy
            if args.policy_class == 'ACT':
                enc_layers = 4
                dec_layers = 7
                nheads = 8
                policy_config = {
                    'lr': 1e-4,
                    'num_queries': 100,
                    'kl_weight': 10,
                    'hidden_dim': 512,
                    'dim_feedforward': 3200,
                    'lr_backbone': 1e-5,
                    'backbone': 'resnet18',
                    'enc_layers': enc_layers,
                    'dec_layers': dec_layers,
                    'nheads': nheads,
                    'camera_names': camera_names,
                }
            else:
                policy_config = {
                    'lr': 1e-4,
                    'lr_backbone': 1e-5,
                    'backbone': 'resnet18',
                    'num_queries': 1,
                    'camera_names': camera_names,
                }
            policy = make_policy(args.policy_class, policy_config)
            policy.load_state_dict(torch.load(ckpt_path, map_location=args.device))
            policy.to(args.device)
            policy.eval()
            # Predict
            preds, _ = evaluate_on_episode(policy, episode_dir, camera_names, stats, args.device)
            preds_over_epochs.append(preds)
        # Now, create video
        video_path = os.path.join(args.output_dir, f"{episode_name}_trajectories_over_epochs.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2
        frame_size = (1600, 900)
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        for i, preds in enumerate(preds_over_epochs):
            fig, axes = plt.subplots(2, 4, figsize=(16, 9))
            axes = axes.flatten()
            for j in range(8):
                ax = axes[j]
                timesteps = range(len(gt))
                ax.plot(timesteps, gt[:, j], 'b-', label='GT', alpha=0.7)
                if len(preds) > 0:
                    ax.plot(timesteps, preds[:, j], 'r--', label='Pred', alpha=0.7)
                ax.set_title(joint_names[j])
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Value')
                if j == 0:
                    ax.legend()
                ax.grid(True, alpha=0.3)
            plt.suptitle(f"{episode_name} - Epoch {epoch_numbers[i]}")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, frame_size)
            out.write(img)
            plt.close(fig)
        out.release()
        print(f"Saved video: {video_path}")

def main():
    parser = argparse.ArgumentParser(description='Sanity check evaluation for Franka robot')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing the trained model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing the training dataset')
    parser.add_argument('--policy_class', type=str, required=True, choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results (default: ckpt_dir/sanity_check)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, 'sanity_check')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get data parameters
    camera_names, total_episodes, episode_len = get_data_params(args.dataset_dir)
    print(f"Found {total_episodes} episodes with {len(camera_names)} cameras: {camera_names}")
    
    # Load model and stats
    print("Loading model and statistics...")
    stats, ckpt_path = load_model_and_stats(args.ckpt_dir, args.policy_class)
    
    # Create policy config (this should match your training config)
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    
    if args.policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': 1e-4,  # This doesn't matter for evaluation
            'num_queries': 100,  # You might need to adjust this based on your training
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
        }
    elif args.policy_class == 'CNNMLP':
        policy_config = {
            'lr': 1e-4,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
        }
    
    # Create and load policy
    print(f"Creating {args.policy_class} policy...")
    policy = make_policy(args.policy_class, policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    policy.to(args.device)
    policy.eval()
    print(f"Loaded model from {ckpt_path}")
    
    # Get episode directories
    episode_dirs = sorted([
        os.path.join(args.dataset_dir, d) 
        for d in os.listdir(args.dataset_dir) 
        if d.startswith('episode_')
    ])
    
    # Limit number of episodes to evaluate
    num_eval_episodes = min(2, len(episode_dirs))
    episode_dirs = episode_dirs[:num_eval_episodes]
    
    print(f"Evaluating on {num_eval_episodes} episodes...")
    
    all_metrics = []
    
    # Evaluate on each episode
    for i, episode_dir in enumerate(tqdm(episode_dirs, desc="Evaluating episodes")):
        episode_name = os.path.basename(episode_dir)
        
        try:
            predictions, ground_truths = evaluate_on_episode(
                policy, episode_dir, camera_names, stats, args.device
            )
            
            if len(predictions) > 0:
                metrics = compute_metrics(predictions, ground_truths)
                metrics['episode'] = episode_name
                all_metrics.append(metrics)
                
                # Save plots
                plot_path = os.path.join(args.output_dir, f'{episode_name}_predictions.png')
                plot_predictions(predictions, ground_truths, plot_path, episode_name)
                
                print(f"{episode_name}: MSE={metrics['overall_mse']:.4f}, MAE={metrics['overall_mae']:.4f}")
            else:
                print(f"Warning: No valid predictions for {episode_name}")
                
        except Exception as e:
            print(f"Error evaluating {episode_name}: {e}")
            continue
    
    # Compute overall statistics
    if all_metrics:
        overall_mse = np.mean([m['overall_mse'] for m in all_metrics])
        overall_mae = np.mean([m['overall_mae'] for m in all_metrics])
        
        print(f"\n{'='*50}")
        print(f"OVERALL RESULTS:")
        print(f"{'='*50}")
        print(f"Average MSE: {overall_mse:.6f}")
        print(f"Average MAE: {overall_mae:.6f}")
        
        # Per-joint statistics
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'gripper']
        print(f"\nPer-joint MAE:")
        for joint in joint_names:
            joint_maes = [m['per_joint_mae'][joint] for m in all_metrics]
            avg_mae = np.mean(joint_maes)
            print(f"  {joint}: {avg_mae:.6f}")
        
        # Save results to file
        results_file = os.path.join(args.output_dir, 'sanity_check_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Sanity Check Evaluation Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Dataset: {args.dataset_dir}\n")
            f.write(f"Model: {ckpt_path}\n")
            f.write(f"Policy: {args.policy_class}\n")
            f.write(f"Episodes evaluated: {len(all_metrics)}\n\n")
            f.write(f"Overall MSE: {overall_mse:.6f}\n")
            f.write(f"Overall MAE: {overall_mae:.6f}\n\n")
            f.write(f"Per-joint MAE:\n")
            for joint in joint_names:
                joint_maes = [m['per_joint_mae'][joint] for m in all_metrics]
                avg_mae = np.mean(joint_maes)
                f.write(f"  {joint}: {avg_mae:.6f}\n")
        
        print(f"\nResults saved to {results_file}")
        print(f"Plots saved to {args.output_dir}")
        
    else:
        print("No successful evaluations completed!")
    
    make_epoch_videos(episode_dirs, camera_names, stats, args)


if __name__ == '__main__':
    main()