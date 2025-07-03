import argparse
import os
from pathlib import Path

from launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(description="Launch learning rate sweep experiments.")
    parser.add_argument("--launch", action="store_true", help="Launch the experiments")
    parser.add_argument("--debug", action="store_true", help="Use debug settings (dev queue, shorter time)")
    args = parser.parse_args()

    return args


cmd_modifiers = []
exps = []

# Learning rate sweep values - covering typical ranges for transformer training
learning_rates = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]

for lr in learning_rates:
    exp_name = f"pomgpt_baseline_lr_{str(lr).replace('.', 'p')}"
    job_name = f"lr_sweep"
    jz_exp = JeanZayExperiment(exp_name, job_name)
    jz_exp.nodes = 1
    jz_exp.num_gpus_per_node = 4
    jz_exp.qos = "dev"
    jz_exp.account = "syq"
    jz_exp.gpu_type = "h100"
    jz_exp.time = "02:00:00"
    jz_exp.cmd_path = "train.py"

    exps.append(jz_exp)

    trainer_modifiers = {
        "experiment_name": exp_name,
    }

    exp_modifier = {
        "experiment": "pomgpt_baseline",
        "training.learning_rate": lr,
        "training.batch_size": 48,
        "training.accumulation": 2,
    }

    cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    args = parse_mode()
    
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        exp.build_cmd(cmd_modifier)
        if args.launch:
            exp.launch(debug=args.debug)
        else:
            print(f"Prepared experiment: {exp.expname}")
            print(f"Command: {exp.cmd}")
            print("---") 