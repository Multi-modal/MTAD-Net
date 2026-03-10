#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
或指定数据集:
  python anomaly_detection-train.py --dataset Abilene
  python anomaly_detection-train.py --dataset Geant
"""

import subprocess
import sys
import os
import torch
import argparse
import json
import traceback


def get_args():
    parser = argparse.ArgumentParser(description='MTAD Anomaly Detection')

    parser.add_argument('--dataset', type=str, default='Abilene', choices=['Abilene', 'Geant'],
                        help='Choose dataset: Abilene or Geant')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 32 for Abilene, 8 for Geant)')

    args = parser.parse_args()

    if args.batch_size is None:
        if args.dataset == 'Abilene':
            args.batch_size = 32
        elif args.dataset == 'Geant':
            args.batch_size = 8

    return args

args = get_args()

def print_banner(text, width=100):
    print("\n" + "=" * width)
    print(text)
    print("=" * width)

args = get_args()
DEFAULT_CONFIG = {
    'dataset': args.dataset,  # 默认使用Abilene数据集
    'eval_mode': 'flow_wise',
    'epochs': 10,
    'batch_size': args.batch_size,#32->8 when test Geant
    'learning_rate': 0.0001,
    'color_space': 'hsv',
    'use_multi_scale': 'False',
    'use_detail_preserve': 'True',
}

DATASET_CONFIGS = {
    'Abilene': {
        'default_path': './dataset/AbileneRobust/',
        'default_flows': 144,
        'description': 'Abilene Network Traffic',
        'data_type': 'Abilene',
        'default_anomaly_ratio': 3
    },
    'Geant': {
        'default_path': './dataset/geant/',
        'default_flows': 529,
        'description': 'Geant Network Traffic',
        'data_type': 'Geant',
        'default_anomaly_ratio': 18
    }
}


def load_dataset_stats(root_path, dataset_name):

    stats_file = os.path.join(root_path, 'dataset_stats.json')
    default_anomaly_ratio = DATASET_CONFIGS[dataset_name]['default_anomaly_ratio']

    stats = {
        'anomaly_ratio': default_anomaly_ratio,
        'num_flows': DATASET_CONFIGS[dataset_name]['default_flows'],
        'train_samples': 'N/A',
        'test_samples': 'N/A',
        'anomaly_points': 'N/A',
        'total_points': 'N/A'
    }

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            loaded_stats = json.load(f)
        stats.update(loaded_stats)
    return stats


def validate_dataset(root_path):

    required_files = ['train.npy', 'test.npy', 'test_label.npy']
    missing_files = []

    for filename in required_files:
        filepath = os.path.join(root_path, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    return missing_files


def main():
    if torch.cuda.is_available():
        gpu_available = True
        num_gpus = torch.cuda.device_count()
    else:
        gpu_available = False
        num_gpus = 0

    parser = argparse.ArgumentParser(
        description='Network Anomaly Detection Training with Defaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 所有参数都有默认值
    parser.add_argument('--dataset', type=str,
                        default=DEFAULT_CONFIG['dataset'],
                        choices=['Abilene', 'Geant'],
                        help='数据集名称')
    parser.add_argument('--root_path', type=str, default=None,
                        help='数据集根目录（默认自动推断）')
    parser.add_argument('--num_vars', type=int, default=None,
                        help='网络流数量（默认从dataset_stats.json读取）')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='输入序列长度')
    parser.add_argument('--color_space', type=str,
                        default=DEFAULT_CONFIG['color_space'],
                        choices=['distinct', 'hsv', 'categorical'],
                        help='颜色空间选择')
    parser.add_argument('--vision_patch_size', type=int, default=16,
                        help='视觉分块大小')
    parser.add_argument('--vision_grid_size', type=str, default='3,3',
                        help='网格尺寸')
    parser.add_argument('--use_multi_scale', type=str,
                        default=DEFAULT_CONFIG['use_multi_scale'],
                        choices=['True', 'False'],
                        help='多尺度融合')
    parser.add_argument('--use_detail_preserve', type=str,
                        default=DEFAULT_CONFIG['use_detail_preserve'],
                        choices=['True', 'False'],
                        help='细节保留增强')
    parser.add_argument('--eval_mode', type=str,
                        default=DEFAULT_CONFIG['eval_mode'],
                        choices=['time_wise', 'flow_wise'],
                        help='评估模式')
    parser.add_argument('--epochs', type=int,
                        default=DEFAULT_CONFIG['epochs'],
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int,
                        default=DEFAULT_CONFIG['batch_size'],
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float,
                        default=DEFAULT_CONFIG['learning_rate'],
                        help='学习率')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='是否使用多GPU')
    parser.add_argument('--save_images', type=str, default='False',
                        choices=['True', 'False'],
                        help='是否保存生成的图像')
    parser.add_argument('--model_id', type=str, default=None,
                        help='模型ID（默认自动生成）')

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_config = DATASET_CONFIGS[dataset_name]
    print(f"\n  Selected Dataset: {dataset_config['description']}")

    # 设置数据集路径
    if args.root_path is None:
        args.root_path = dataset_config['default_path']
    stats = load_dataset_stats(args.root_path, dataset_name)

    if args.num_vars is None:
        args.num_vars = stats.get('num_flows', dataset_config['default_flows'])
    anomaly_ratio = stats.get('anomaly_ratio')


    if args.model_id is None:
        args.model_id = (
            f"{dataset_name}_{args.eval_mode}_"
            f"{args.color_space}_ms{args.use_multi_scale}_dp{args.use_detail_preserve}"
        )

    use_gpu = gpu_available
    use_multi_gpu = args.use_multi_gpu and gpu_available and num_gpus > 1

    cmd = [
        sys.executable, "run.py",
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--model", "MTAD",
        "--model_id", args.model_id,
        "--data", dataset_config['data_type'],
        "--root_path", args.root_path,
        "--data_path", "train",
        "--features", "M",
        "--freq", "h",
        "--target", "OT",
        "--seq_len", str(args.seq_len),
        "--label_len", "0",
        "--pred_len", "100",
        "--enc_in", str(args.num_vars),
        "--dec_in", str(args.num_vars),
        "--c_out", str(args.num_vars),
        "--d_model", "128",
        "--n_heads", "8",
        "--e_layers", "2",
        "--d_layers", "1",
        "--d_ff", "768",
        "--dropout", "0.1",
        "--activation", "gelu",
        "--embed", "timeF",
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_workers", "0",
        "--train_epochs", str(args.epochs),
        "--patience", "5",
        "--itr", "1",
        "--lradj", "type1",
        "--des", "Exp",
        # "--gradient_clip", str(args.gradient_clip),
        # "--warmup_epochs", str(args.warmup_epochs),
        # "--weight_decay", str(args.weight_decay),
        "--anomaly_ratio", str(anomaly_ratio),
        # "--eval_mode", args.eval_mode,
        "--mtad_type", "CLIP",
        "--image_size", "224",
        "--patch_len", "16",
        "--stride", "8",
        "--padding", "8",
        "--patch_memory_size", "100",
        "--norm_const", "0.4",
        "--top_k", "5",
        "--finetune_mtad", "False",
        "--use_mem_gate", "False",
        # "--color_space", args.color_space,
        # "--vision_patch_size", str(args.vision_patch_size),
        # "--vision_grid_size", args.vision_grid_size,
        # "--use_multi_scale", args.use_multi_scale,
        # "--use_detail_preserve", args.use_detail_preserve,
        "--save_images", args.save_images,
        "--use_gpu", str(use_gpu),
        "--gpu", str(args.gpu),
        "--seed", "2024",
        "--checkpoints", "./checkpoints/",
    ]

    if use_multi_gpu:
        cmd.extend(["--use_multi_gpu", "--devices", "0,1"])

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    except KeyboardInterrupt:
        print_banner("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        print_banner(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()