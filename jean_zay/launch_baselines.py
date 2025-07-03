import argparse
import os
from pathlib import Path

from launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    return args


cmd_modifiers = []
exps = []

models = {
    "transformers_baseline": "transformers_baseline",
    "pomgpt_baseline": "pomgpt_baseline",
}
for model_name, model_override in models.items():
        exp_name = model_name
        job_name = f"baselines"
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
            "experiment": model_override,
            "training.batch_size": 48,
            "training.accumulation": 2,
        }

        cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
