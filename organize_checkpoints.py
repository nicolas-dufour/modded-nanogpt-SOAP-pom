#!/usr/bin/env python3
"""
Utility script to organize checkpoint files for the Streamlit demo.
This script helps copy and organize trained models from the outputs/ directory
into a more organized checkpoints/ structure.
"""

import os
import shutil
import argparse
from pathlib import Path
import torch
from datetime import datetime


def get_checkpoint_info(checkpoint_path):
    """Extract basic info from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'step': checkpoint.get('step', 'unknown'),
            'experiment': 'unknown',
            'config': None
        }
        
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            info['experiment'] = getattr(cfg, 'experiment_name', 'unknown')
            info['config'] = cfg
            
        return info
    except Exception as e:
        print(f"Warning: Could not read {checkpoint_path}: {e}")
        return None


def find_output_checkpoints(outputs_dir="outputs"):
    """Find all checkpoint files in the outputs directory."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"Outputs directory {outputs_dir} does not exist.")
        return []
    
    checkpoints = []
    for checkpoint_file in outputs_path.glob("**/*.pt"):
        info = get_checkpoint_info(checkpoint_file)
        if info:
            checkpoints.append((checkpoint_file, info))
    
    return checkpoints


def organize_checkpoints(dry_run=False, checkpoints_dir="checkpoints"):
    """Organize checkpoints from outputs/ into checkpoints/ directory."""
    
    checkpoints_path = Path(checkpoints_dir)
    checkpoints_path.mkdir(exist_ok=True)
    
    print(f"🔍 Scanning for checkpoints in outputs/...")
    output_checkpoints = find_output_checkpoints()
    
    if not output_checkpoints:
        print("❌ No checkpoint files found in outputs/")
        return
    
    print(f"✅ Found {len(output_checkpoints)} checkpoint files")
    
    # Group by experiment
    experiments = {}
    for checkpoint_path, info in output_checkpoints:
        exp_name = info['experiment']
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append((checkpoint_path, info))
    
    print(f"\n📊 Found {len(experiments)} experiments:")
    for exp_name, checkpoints in experiments.items():
        print(f"  - {exp_name}: {len(checkpoints)} checkpoints")
    
    print(f"\n{'🎯 Would organize' if dry_run else '📁 Organizing'} checkpoints:")
    
    for exp_name, checkpoints in experiments.items():
        # Create experiment directory
        exp_dir = checkpoints_path / exp_name
        
        if not dry_run:
            exp_dir.mkdir(exist_ok=True)
        
        print(f"\n📂 {exp_name}/")
        
        # Sort checkpoints by step
        checkpoints.sort(key=lambda x: x[1]['step'] if isinstance(x[1]['step'], int) else 0)
        
        for checkpoint_path, info in checkpoints:
            step = info['step']
            
            # Create a descriptive filename
            if isinstance(step, int):
                new_filename = f"checkpoint_step_{step:06d}.pt"
            else:
                # Use original filename if step is unknown
                new_filename = checkpoint_path.name
            
            dest_path = exp_dir / new_filename
            
            print(f"  ├── {new_filename}")
            print(f"  │   └── Step: {step}, Size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
            
            if not dry_run:
                if dest_path.exists():
                    print(f"  │   ⚠️  File exists, skipping...")
                else:
                    shutil.copy2(checkpoint_path, dest_path)
                    print(f"  │   ✅ Copied successfully")
    
    if dry_run:
        print(f"\n💡 This was a dry run. Use --execute to actually copy files.")
        print(f"💡 Files will be copied to: {checkpoints_path.absolute()}")
    else:
        print(f"\n🎉 Organization complete! Checkpoints are now in: {checkpoints_path.absolute()}")
        print(f"🚀 You can now use these in the Streamlit demo!")


def list_checkpoints(checkpoints_dir="checkpoints"):
    """List all organized checkpoints."""
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"❌ Checkpoints directory {checkpoints_dir} does not exist.")
        print(f"💡 Run with --organize to create it from outputs/")
        return
    
    print(f"📁 Checkpoints in {checkpoints_path.absolute()}:")
    
    checkpoint_files = list(checkpoints_path.glob("**/*.pt"))
    if not checkpoint_files:
        print("❌ No checkpoint files found.")
        print("💡 Run with --organize to copy from outputs/")
        return
    
    # Group by directory
    by_dir = {}
    for cp_file in checkpoint_files:
        if cp_file.parent == checkpoints_path:
            dir_name = "root"
        else:
            dir_name = cp_file.relative_to(checkpoints_path).parts[0]
        
        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(cp_file)
    
    for dir_name, files in by_dir.items():
        if dir_name == "root":
            print(f"\n📂 / (root)")
        else:
            print(f"\n📂 {dir_name}/")
        
        for cp_file in sorted(files):
            info = get_checkpoint_info(cp_file)
            if info:
                step = info['step']
                size_mb = cp_file.stat().st_size / (1024**2)
                modified = datetime.fromtimestamp(cp_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                print(f"  ├── {cp_file.name}")
                print(f"  │   ├── Step: {step}")
                print(f"  │   ├── Size: {size_mb:.1f} MB")
                print(f"  │   └── Modified: {modified}")
            else:
                print(f"  ├── {cp_file.name} (could not read info)")


def main():
    parser = argparse.ArgumentParser(description="Organize checkpoint files for the Streamlit demo")
    parser.add_argument(
        "--organize", 
        action="store_true", 
        help="Organize checkpoints from outputs/ to checkpoints/"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List organized checkpoints"
    )
    parser.add_argument(
        "--execute", 
        action="store_true", 
        help="Actually perform the organization (default is dry run)"
    )
    parser.add_argument(
        "--checkpoints-dir", 
        default="checkpoints", 
        help="Checkpoints directory (default: checkpoints)"
    )
    parser.add_argument(
        "--outputs-dir", 
        default="outputs", 
        help="Outputs directory to scan (default: outputs)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.checkpoints_dir)
    elif args.organize:
        organize_checkpoints(dry_run=not args.execute, checkpoints_dir=args.checkpoints_dir)
    else:
        print("🤖 Checkpoint Organizer for Modded NanoGPT Demo")
        print("=" * 50)
        print()
        print("Usage:")
        print("  --organize          Organize checkpoints from outputs/")
        print("  --organize --execute Actually perform the organization")
        print("  --list              List organized checkpoints")
        print()
        print("Examples:")
        print("  python organize_checkpoints.py --organize")
        print("  python organize_checkpoints.py --organize --execute")
        print("  python organize_checkpoints.py --list")
        print()
        print("💡 By default, --organize does a dry run. Use --execute to actually copy files.")


if __name__ == "__main__":
    main() 