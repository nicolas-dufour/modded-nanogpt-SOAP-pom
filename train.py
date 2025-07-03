import os
import sys
import uuid
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
import tiktoken
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, ProgressColumn
from rich.console import Console
from rich.text import Text
from count_params import count_parameters
from data import DistributedDataLoader

import os
os.environ["TIKTOKEN_CACHE_DIR"] = ".tiktoken_cache"
Path(".tiktoken_cache").mkdir(parents=True, exist_ok=True)
# Set float32 matmul precision to match reference implementation
torch.set_float32_matmul_precision('high')

class SpeedColumn(ProgressColumn):
    """Custom column to display training speed"""
    
    def __init__(self):
        super().__init__()
        self.last_time = time.time()
        self.last_step = 0
        self.speed = 0.0
    
    def render(self, task):
        current_time = time.time()
        current_step = task.completed
        
        # Update speed calculation every second
        if current_time - self.last_time >= 1.0 and current_step > self.last_step:
            self.speed = (current_step - self.last_step) / (current_time - self.last_time)
            self.last_time = current_time
            self.last_step = current_step
        
        return Text(f"{self.speed:.2f} it/s", style="bold yellow")

def print0(*args, **kwargs):
    """Modified print that only prints from the master process."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def get_lr(step: int, num_iterations: int, warmup_iters: int, warmdown_iters: int) -> float:
    """
    Calculate learning rate scale for current step.
    
    Args:
        step: Current training step
        num_iterations: Total number of iterations
        warmup_iters: Number of warmup iterations
        warmdown_iters: Number of warmdown iterations
        
    Returns:
        Learning rate scale
    """
    assert step <= num_iterations
    
    # 1) linear warmup for warmup_iters steps
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    # 2) constant lr for a while
    elif step < num_iterations - warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - step) / warmdown_iters
        return decay_ratio


def sample_from_model(model, prompt_tokens=None, max_new_tokens=64, temperature=1.0, top_k=50, device='cuda', generator=None):
    """
    Sample text from the model.
    
    Args:
        model: The GPT model
        prompt_tokens: Optional tensor of prompt tokens. If None, starts with EOT token.
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Number of top tokens to sample from
        device: Device to run on
        generator: torch.Generator for reproducible sampling
        
    Returns:
        Tensor of generated tokens including prompt
    """
    # Get the original model (unwrap from compilation if needed)
    original_model = model
    if hasattr(model, '_orig_mod'):
        # If the model is compiled, use the original uncompiled version for sampling
        original_model = model._orig_mod
    
    original_model.eval()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    with torch.no_grad():
        # Start with prompt or EOT token
        if prompt_tokens is None:
            # Start with end of text token for unconditional generation
            tokens = torch.tensor([[eot]], dtype=torch.long, device=device)
        else:
            tokens = prompt_tokens.clone().contiguous()  # Ensure contiguous memory layout
        
        for _ in range(max_new_tokens):
            # Ensure tokens are contiguous before forward pass
            tokens = tokens.contiguous()
            
            # Forward pass with original (uncompiled) model
            logits, _ = original_model(tokens, targets=None, return_logits=True)
            logits = logits[:, -1, :] / temperature  # Take last token and apply temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            
            # Append to sequence and ensure contiguous layout
            tokens = torch.cat([tokens, next_token], dim=1).contiguous()
            
            # Stop if we generate EOT token (except for the first token)
            if tokens.shape[1] > 1 and next_token.item() == eot:
                break
    
    # Clear GPU memory cache after sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Restore training mode for the original model
    original_model.train()
    return tokens


def generate_samples(model, val_loader, device='cuda', num_unconditional=3, num_completions=3, 
                    max_new_tokens=64, temperature=1.0, top_k=50, prompt_length=32, base_seed=42, debug=False):
    """
    Generate text samples for logging.
    
    Args:
        model: The GPT model
        val_loader: Validation data loader
        device: Device to run on
        num_unconditional: Number of unconditional samples to generate
        num_completions: Number of completion samples to generate
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Number of top tokens to sample from
        prompt_length: Length of prompt for completion samples
        base_seed: Base seed for reproducible sampling
        
    Returns:
        Dictionary with 'unconditional' and 'completions' lists
    """
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    samples = {'unconditional': [], 'completions': []}
    
    # Generate unconditional samples with consistent seeds
    for i in range(num_unconditional):
        # Create generator with specific seed for each unconditional sample
        generator = torch.Generator(device=device)
        generator.manual_seed(base_seed + i)
        
        tokens = sample_from_model(
            model, 
            prompt_tokens=None, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            generator=generator
        )
        text = enc.decode(tokens[0].cpu().numpy())
        samples['unconditional'].append(text)
    
    # Generate completion samples from validation set
    # Reset validation loader to ensure we always get the same first batch
    val_loader.reset()
    val_x, _ = val_loader.next_batch()
    
    # Debug: Print first few tokens of first sample to verify consistency
    if debug and len(val_x) > 0:
        first_tokens = val_x[0, :min(8, val_x.shape[1])].cpu().tolist()
        print(f"Debug - First validation tokens: {first_tokens}")
    
    # Use consistent sample indices for reproducible prompts
    completion_indices = list(range(min(num_completions, val_x.shape[0])))
    
    for i in completion_indices:
        # Take first prompt_length tokens as prompt
        prompt_tokens = val_x[i:i+1, :prompt_length]  # Keep batch dimension
        
        # Create generator with specific seed for each completion sample (offset from unconditional seeds)
        generator = torch.Generator(device=device)
        generator.manual_seed(base_seed + 1000 + i)
        
        # Generate completion
        completion_tokens = sample_from_model(
            model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            generator=generator
        )
        
        # Decode prompt and full completion separately for better display
        prompt_text = enc.decode(prompt_tokens[0].cpu().numpy())
        full_text = enc.decode(completion_tokens[0].cpu().numpy())
        completion_text = full_text[len(prompt_text):]
        
        samples['completions'].append({
            'prompt': prompt_text,
            'completion': completion_text,
            'full': full_text
        })
    
    # Clean up memory after all sampling is complete
    del val_x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return samples


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print0(f"Running pytorch {torch.version.__version__}")
    print0(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Set up distributed training
    assert torch.cuda.is_available()
    dist.init_process_group(backend=cfg.distributed.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0)
    
    # Set up wandb (only on rank 0)
    if master_process:
        wandb.init(project="pom_archi", entity="imaginelab", config=dict_cfg, name=cfg.experiment_name)
    
    # Load data
    train_loader = DistributedDataLoader(
        cfg.data.train.input_bin,
        cfg.training.batch_size,
        cfg.training.sequence_length,
        ddp_rank,
        ddp_world_size
    )
    print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    
    val_loader = DistributedDataLoader(
        cfg.data.val.input_bin,
        cfg.training.batch_size,
        cfg.training.sequence_length,
        ddp_rank,
        ddp_world_size
    )
    print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    
    x, y = train_loader.next_batch()
    
    # Initialize model using Hydra instantiate
    model = instantiate(cfg.model.gpt)
    model = model.cuda()
    
    # Count and display model parameters
    total_params, trainable_params = count_parameters(model, print_breakdown=True)
    
    # Log parameter counts to wandb
    if master_process:
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/memory_mb_fp32": total_params * 4 / 1024**2,
            "model/memory_mb_bf16": total_params * 2 / 1024**2
        })
    
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    
    print0("compiling the model...")
    if cfg.hardware.compile:
        model = torch.compile(model)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=cfg.distributed.find_unused_parameters)
    raw_model = model.module
    
    # Set up context manager for mixed precision
    ctx = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, cfg.hardware.dtype))
    
    # Initialize optimizer using model's configure_optimizers method
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,  # Use config value (0.5) to match reference
        learning_rate=cfg.training.learning_rate,
        betas=(0.9, 0.95)  # Fixed betas to match reference implementation
    )
    
    # Set up logging
    run_id = str(uuid.uuid4())
    if master_process:
        log_dir = Path(cfg.logging.log_dir) / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / "log.txt"
        logfile.touch()  # create empty log file
        
        # Set up checkpoint directory
        checkpoint_dir = Path("checkpoints") / cfg.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up rich progress bar (only for master process)
    console = Console()
    if master_process:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Training", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            SpeedColumn(),
            "•",
            TimeElapsedColumn(),
            "•", 
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        )
        progress.start()
        task = progress.add_task("Training", total=cfg.training.num_iterations)
    
    # Training loop
    for step in range(cfg.training.num_iterations + 1):
        last_step = (step == cfg.training.num_iterations)
        
        # Validation
        if (last_step or (cfg.evaluation.val_loss_every > 0 and step % cfg.evaluation.val_loss_every == 0)):
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(cfg.evaluation.val_max_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= cfg.evaluation.val_max_steps
            
            # Generate text samples (only on master process to avoid duplicate generations)
            if master_process:
                val_loader.reset()  # Reset val loader for sampling
                
                # Configure sampling parameters - can be added to config later
                sample_every = getattr(cfg.evaluation, 'sample_every', cfg.evaluation.val_loss_every)
                should_sample = (step % sample_every == 0) or last_step
                
                if should_sample:
                    print0("Generating text samples...")
                    # Reset validation loader to ensure consistent prompts across steps
                    val_loader.reset()
                    # Force the loader to the beginning and get a consistent state
                    val_loader.current_shard = 0
                    val_loader.current_position = val_loader.process_rank * val_loader.B * val_loader.T
                    
                    samples = generate_samples(
                        raw_model,  # Use raw model without DDP wrapper
                        val_loader,
                        device=device,
                        num_unconditional=getattr(cfg.evaluation, 'num_unconditional_samples', 3),
                        num_completions=getattr(cfg.evaluation, 'num_completion_samples', 3),
                        max_new_tokens=getattr(cfg.evaluation, 'max_new_tokens', 64),
                        temperature=getattr(cfg.evaluation, 'temperature', 1.0),
                        top_k=getattr(cfg.evaluation, 'top_k', 50),
                        prompt_length=getattr(cfg.evaluation, 'prompt_length', 32),
                        base_seed=getattr(cfg.evaluation, 'sample_seed', 42),
                        debug=getattr(cfg.evaluation, 'debug_sampling', False)
                    )
                    
                    # Log samples to console
                    print0("\n" + "="*60)
                    print0(f"TEXT SAMPLES AT STEP {step}")
                    print0("="*60)
                    
                    print0("\nUNCONDITIONAL SAMPLES:")
                    print0("-"*40)
                    for i, text in enumerate(samples['unconditional']):
                        print0(f"Sample {i+1}:")
                        print0(repr(text))  # Use repr to show escape chars
                        print0("")
                    
                    print0("\nCOMPLETION SAMPLES:")
                    print0("-"*40)
                    for i, sample in enumerate(samples['completions']):
                        print0(f"Completion {i+1}:")
                        print0(f"Prompt: {repr(sample['prompt'])}")
                        print0(f"Generated: {repr(sample['completion'])}")
                        print0("")
                    
                    # Log to wandb
                    wandb_log = {
                        "val_loss": val_loss.item(), 
                        "step": step
                    }
                    
                    # Add text samples to wandb
                    for i, text in enumerate(samples['unconditional']):
                        wandb_log[f"unconditional_sample_{i+1}"] = text
                    
                    for i, sample in enumerate(samples['completions']):
                        wandb_log[f"completion_prompt_{i+1}"] = sample['prompt']
                        wandb_log[f"completion_generated_{i+1}"] = sample['completion']
                    
                    wandb.log(wandb_log)
                    
                    print0("="*60)
                    
                    # Free memory after sampling
                    del samples
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    # Just log validation loss
                    wandb.log({"val_loss": val_loss.item(), "step": step})
            
            print0(f"val loss {val_loss:.4f}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} tel:{val_loss}\n")
        
        # Save checkpoint
        if master_process and (last_step or (cfg.evaluation.save_every > 0 and step % cfg.evaluation.save_every == 0)):
            checkpoint = {
                'step': step,
                'config': cfg,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_dir / "last.ckpt")
        
        if last_step:
            break
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Training
        model.train()
        for _ in range(cfg.training.accumulation):
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            x, y = train_loader.next_batch()
            loss.backward()
        
        # Gradient accumulation
        for p in model.parameters():
            p.grad /= cfg.training.accumulation
        
        # Learning rate scheduling
        lr_scale = get_lr(
            step,
            cfg.training.num_iterations,
            cfg.training.warmup_iters,
            cfg.training.warmdown_iters
        )
        optimizer.scale_lrs(lr_scale)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        # Logging
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        tokens_per_second = ddp_world_size * cfg.training.batch_size * cfg.training.sequence_length / (t1 - t0)
        
        # Log training loss to wandb
        if master_process:
            wandb.log({
                "train_loss": train_loss.item(),
                "lr_scale": lr_scale,
                "tokens_per_second": tokens_per_second,
                "step": step + 1
            })
        
        # Update progress bar
        if master_process:
            progress.update(task, advance=1, description=f"[bold blue]Training (loss: {train_loss.item():.4f}, lr: {lr_scale:.2e})")
        
        # Detailed logging every N steps
        if step % cfg.logging.log_every == 0:
            print0(f"step {step+1:4d}/{cfg.training.num_iterations} | "
                   f"train loss {train_loss.item():.4f} | "
                   f"lr_scale {lr_scale:.2e} | "
                   f"({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trl:{train_loss.item()}\n")
    
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    
    # Close progress bar
    if master_process:
        progress.stop()
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main() 