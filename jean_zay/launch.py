import os
from pathlib import Path
import math
import shlex


class JeanZayExperiment:
    def __init__(
        self,
        exp_name,
        job_name,
        slurm_array_nb_jobs=None,
        max_simultaneous_jobs=None,
        sub_job_index=0,
        num_nodes=1,
        num_gpus_per_node=1,
        qos="t3",
        account="syq",
        gpu_type="a100",
        cmd_path="train.py",
        time=None,
        launch_from_compute_node=False,
        min_time=None,
    ):
        self.expname = exp_name
        self.job_name = job_name
        self.nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.qos = qos
        self.account = account
        self.gpu_type = gpu_type
        self.slurm_array_nb_jobs = slurm_array_nb_jobs
        self.max_simultaneous_jobs = max_simultaneous_jobs
        self.sub_job_index = sub_job_index
        self.cmd_path = cmd_path
        self.time = time
        self.min_time = min_time
        self.launch_from_compute_node = launch_from_compute_node

        # Initialize paths
        self.slurm_script_path = None
        self.slurm_out_path = None
        self.slurm_err_path = None
        self.slurm_job_id = None  # To store job ID after submission
        self.cmd = None  # For single command
        self.cmds = None  # For list of commands

    def build_cmd(self, hydra_args=None, hydra_arg_list=None):
        # Validate input: exactly one of hydra_args or hydra_arg_list must be provided
        if (hydra_args is None and hydra_arg_list is None) or (
            hydra_args is not None and hydra_arg_list is not None
        ):
            raise ValueError(
                "Provide exactly one of hydra_args (for single command or numeric array) or hydra_arg_list (for command list array)."
            )

        if hydra_arg_list is not None:
            if self.slurm_array_nb_jobs is not None:
                raise ValueError(
                    "Cannot provide slurm_array_nb_jobs during initialization when using hydra_arg_list in build_cmd."
                )
            self.cmds = []
            for args_dict in hydra_arg_list:
                hydra_modifiers = []
                for hydra_arg, value in args_dict.items():
                    if hydra_arg.startswith("--"):
                        hydra_modifiers.append(f" {hydra_arg} {value}")
                    else:
                        hydra_modifiers.append(f" {hydra_arg}={value}")
                command = f"torchrun --standalone --nproc_per_node={self.num_gpus_per_node} {self.cmd_path} {''.join(hydra_modifiers)}"
                self.cmds.append(command)
                print(f"Prepared command: {command}")
            # Set slurm_array_nb_jobs based on the list length
            self.slurm_array_nb_jobs = len(self.cmds)
            if not self.slurm_array_nb_jobs:
                raise ValueError("hydra_arg_list cannot be empty.")
            print(f"Built {self.slurm_array_nb_jobs} commands for job array.")

        elif hydra_args is not None:
            # Existing logic for single command or numeric array
            # If slurm_array_nb_jobs is provided in init, it defines a numeric array
            # If slurm_array_nb_jobs is None, it's a single command job
            if self.slurm_array_nb_jobs is not None and not isinstance(
                self.slurm_array_nb_jobs, int
            ):
                raise ValueError(
                    "If hydra_args is provided, slurm_array_nb_jobs (from init) must be an integer or None."
                )
            hydra_modifiers = []
            for hydra_arg, value in hydra_args.items():
                if hydra_arg.startswith("--"):
                    hydra_modifiers.append(f" {hydra_arg} {value}")
                else:
                    hydra_modifiers.append(f" {hydra_arg}={value}")
            self.cmd = f"torchrun --standalone --nproc_per_node={self.num_gpus_per_node} {self.cmd_path} {''.join(hydra_modifiers)}"
            if self.slurm_array_nb_jobs is not None:
                print(
                    f"Built base command for numeric array ({self.slurm_array_nb_jobs} jobs): {self.cmd}"
                )
            else:
                print(f"Built single command: {self.cmd}")

    def launch(self, debug=False):
        if debug:
            self.qos = "dev"
            self.time = "01:00:00"
            self.min_time = None
        # Check if either a single command or a list of commands has been built
        if not hasattr(self, "cmd") and not hasattr(self, "cmds"):
            raise ValueError("Run build_cmd first")
        if self.cmd is None and self.cmds is None:
            raise ValueError("Run build_cmd first - no command generated.")

        slurm_partition_directive = ""
        slurm_qos_directive = ""
        slurm_gpu_directive = ""
        slurm_account_directive = ""
        cpus_per_task = 1  # Default value
        module_load_directive = ""
        self.max_array_size = 10000  # Default max array size

        if self.qos == "prepost":
            slurm_partition_directive = f"#SBATCH --partition=prepost"
            self.time = "19:59:59" if self.time is None else self.time
            slurm_gpu_directive = ""  # No specific GPU constraint for prepost
            slurm_account_directive = f"#SBATCH --account={self.account}@a100"  # Account may not be GPU specific
            cpus_per_task = 1  # Assuming 1 CPU for prepost tasks
            module_load_directive = ""  # No specific module load for prepost
            print(f"Launching on partition prepost")

        else:  # Handle GPU partitions
            if self.qos == "t4":
                qos_name = "qos_gpu-t4"
                self.time = "99:59:59" if self.time is None else self.time
            elif self.qos == "t3":
                qos_name = "qos_gpu-t3"
                self.time = "19:59:59" if self.time is None else self.time
            elif self.qos == "dev":
                qos_name = "qos_gpu-dev"
                self.time = "01:59:59" if self.time is None else self.time
            else:
                raise ValueError(f"Not a valid QoS for GPU partitions: {self.qos}")

            if self.gpu_type == "a100":
                slurm_gpu_directive = "#SBATCH -C a100"
                cpus_per_task = 8
                qos_name = qos_name.replace("gpu", "gpu_a100")
                self.max_array_size = 10000
                module_load_directive = "module load arch/a100"
            elif self.gpu_type == "h100":
                slurm_gpu_directive = "#SBATCH -C h100"
                cpus_per_task = 24
                qos_name = qos_name.replace("gpu", "gpu_h100")
                self.max_array_size = 10000
                module_load_directive = "module load arch/h100"
            elif self.gpu_type == "v100":
                slurm_gpu_directive = "#SBATCH -C v100-32g"
                cpus_per_task = 10
                self.max_array_size = 10000
                # No specific module load needed for v100 on Jean Zay ? Check documentation
            else:
                raise ValueError(
                    f"Not a valid GPU type for QoS {self.qos}: {self.gpu_type}"
                )

            slurm_qos_directive = f"#SBATCH --qos={qos_name}"
            slurm_account_directive = (
                f"#SBATCH --account={self.account}@{self.gpu_type}"
            )
            print(f"Launching on qos {qos_name}")

        local_slurmfolder = Path("slurm_logs") / Path(self.expname)
        local_slurmfolder.mkdir(parents=True, exist_ok=True)

        array_string = ""
        job_suffix = ""

        if isinstance(self.slurm_array_nb_jobs, int):
            total_jobs = self.slurm_array_nb_jobs
            if total_jobs <= 0:
                array_string = ""
            elif total_jobs > self.max_array_size:
                num_chunks = math.ceil(total_jobs / self.max_array_size)
                if not (0 <= self.sub_job_index < num_chunks):
                    raise ValueError(
                        f"sub_job_index ({self.sub_job_index}) out of range for {num_chunks} chunks."
                    )

                start_index = self.sub_job_index * self.max_array_size
                end_index = (
                    min((self.sub_job_index + 1) * self.max_array_size, total_jobs) - 1
                )
                array_string = f"{start_index}-{end_index}"
                job_suffix = f"_chunk{self.sub_job_index}"
                print(
                    f"Launching chunk {self.sub_job_index}: jobs {start_index}-{end_index}"
                )
            else:
                if self.sub_job_index != 0:
                    print(
                        f"Warning: sub_job_index is {self.sub_job_index} but only one chunk is needed. Using index 0."
                    )
                    self.sub_job_index = 0
                array_string = f"0-{total_jobs - 1}"

        elif self.slurm_array_nb_jobs is not None:
            raise ValueError("slurm_array_nb_jobs must be an int or None.")

        if array_string and self.max_simultaneous_jobs is not None:
            array_string += f"%{self.max_simultaneous_jobs}"

        sbatch_array = f"#SBATCH --array={array_string}" if array_string else ""

        current_job_name = f"{self.job_name}{job_suffix}"
        slurm_path = local_slurmfolder / f"job_file{job_suffix}.slurm"

        # Store the script path
        self.slurm_script_path = slurm_path

        # Construct and store output/error paths with separate directories per task
        slurm_output_base_dir = f"/lustre/fswork/projects/rech/syq/uey53ph/modded-nanogpt-SOAP-pom/{local_slurmfolder}/job_%j{job_suffix}"
        # Use %a for array task ID to create per-task subdirectories
        self.slurm_out_path = f"{slurm_output_base_dir}/task_%a/std.out"
        self.slurm_err_path = f"{slurm_output_base_dir}/task_%a/std.err"

        # Prepare commands for the SLURM script
        srun_command_line = ""
        bash_definitions = ""
        if self.cmds:
            # Create a bash array definition, quoting each command safely
            quoted_cmds = [shlex.quote(cmd) for cmd in self.cmds]
            bash_definitions = f"CMDS=({' '.join(quoted_cmds)})"
            # Use the SLURM_ARRAY_TASK_ID to index into the bash array
            # Adjust task ID based on the start index of the chunk
            start_index = 0
            if self.slurm_array_nb_jobs > self.max_array_size:
                # Ensure sub_job_index and max_array_size are defined and valid before using
                if self.sub_job_index is None or self.max_array_size is None:
                    raise ValueError(
                        "sub_job_index and max_array_size must be set for chunked arrays."
                    )
                start_index = self.sub_job_index * self.max_array_size

            srun_command_line = (
                f"COMMAND_INDEX=$((SLURM_ARRAY_TASK_ID - {start_index}))\n"
                f'echo "Running command index $COMMAND_INDEX: ${{CMDS[$COMMAND_INDEX]}}"\n'
                f"${{CMDS[$COMMAND_INDEX]}}"
            )

        elif self.cmd:
            # Existing single command execution
            srun_command_line = f"{self.cmd}"
        else:
            # This case should ideally not be reached if build_cmd was called
            raise ValueError(
                "No command or command list was built. Call build_cmd first."
            )

        slurm = f"""#!/bin/bash
#SBATCH --job-name={current_job_name}
{sbatch_array}
#SBATCH --nodes={self.nodes}# number of nodes
{slurm_account_directive}
#SBATCH --ntasks-per-node={self.num_gpus_per_node if self.qos != "prepost" else 1} # Adjust ntasks for prepost
{f'#SBATCH --gres=gpu:{self.num_gpus_per_node}' if self.qos != "prepost" else ""} # No gres for prepost
{slurm_qos_directive}
{slurm_partition_directive}
{slurm_gpu_directive}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --time={self.time}
{"#SBATCH --time-min=" + self.min_time if self.min_time is not None else ""}
#SBATCH --output={self.slurm_out_path}
#SBATCH --error={self.slurm_err_path}
#SBATCH --signal=SIGUSR1@90
module purge
{module_load_directive}
source /lustre/fswork/projects/rech/syq/uey53ph/.venvs/modded_gpt/bin/activate

export PYTHONPATH=/lustre/fswork/projects/rech/syq/uey53ph/.venvs/modded_gpt/bin/activate
export TRANSFORMERS_OFFLINE=1 # to avoid downloading
export HYDRA_FULL_ERROR=1 # to have the full traceback
export WANDB_CACHE_DIR=$NEWSCRATCH/wandb_cache
export TMPDIR=$JOBSCRATCH
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export IS_CLUSTER=True
{f"unset SLURM_CPU_BIND" if self.launch_from_compute_node else ""}

# Define commands if using command list
{bash_definitions}

set -x
# Execute the appropriate command
{srun_command_line}
        """
        with open(slurm_path, "w") as slurm_file:
            slurm_file.write(slurm)
        # if self.launch_from_compute_node:
        #     os.system('unset $(env | egrep "SLURM_|SBATCH_"| cut -d= -f1)')
        print(f"Submitting SLURM script: {self.slurm_script_path}")
        # Capture sbatch output to potentially get job ID
        result = os.popen(f"sbatch {self.slurm_script_path}").read()
        print(result)  # Print sbatch output (e.g., "Submitted batch job 12345")
        try:
            # Attempt to parse job ID
            self.slurm_job_id = int(result.strip().split()[-1])
            print(f"SLURM Job ID: {self.slurm_job_id}")
        except (IndexError, ValueError):
            print("Could not parse SLURM job ID from sbatch output.")
