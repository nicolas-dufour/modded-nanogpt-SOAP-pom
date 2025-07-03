import streamlit as st
import torch
import tiktoken
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import time
import glob
from datetime import datetime

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from models.gpt import GPT
import models.pom as pom

# Configuration
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUTS_DIR = Path("outputs")


def sample_from_model(model, prompt_tokens=None, max_new_tokens=64, temperature=1.0, top_k=50, device='cuda', generator=None):
    """
    Sample text from the model - adapted from train.py
    """
    # Get the original model (unwrap from compilation if needed)
    original_model = model
    if hasattr(model, '_orig_mod'):
        original_model = model._orig_mod
    
    original_model.eval()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    with torch.no_grad():
        # Start with prompt or EOT token
        if prompt_tokens is None:
            tokens = torch.tensor([[eot]], dtype=torch.long, device=device)
        else:
            tokens = prompt_tokens.clone().contiguous()
        
        for _ in range(max_new_tokens):
            tokens = tokens.contiguous()
            
            # Forward pass with original model
            logits, _ = original_model(tokens, targets=None, return_logits=True)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1).contiguous()
            
            # Stop if we generate EOT token (except for the first token)
            if tokens.shape[1] > 1 and next_token.item() == eot:
                break
    
    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    original_model.train()
    return tokens


def find_checkpoint_files():
    """
    Find all checkpoint files in the checkpoints directory and outputs directory.
    Returns a list of (display_name, file_path) tuples.
    """
    checkpoint_files = []
    
    # Check the dedicated checkpoints directory
    if CHECKPOINTS_DIR.exists():
        for checkpoint_file in CHECKPOINTS_DIR.glob("**/*.pt"):
            rel_path = checkpoint_file.relative_to(CHECKPOINTS_DIR)
            display_name = f"checkpoints/{rel_path}"
            checkpoint_files.append((display_name, str(checkpoint_file)))
    
    # Check the outputs directory for training runs
    if OUTPUTS_DIR.exists():
        for checkpoint_file in OUTPUTS_DIR.glob("**/*.pt"):
            # Get the relative path from outputs
            rel_path = checkpoint_file.relative_to(OUTPUTS_DIR)
            # Extract date and time from path for better display
            parts = rel_path.parts
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1] 
                filename = parts[-1]
                display_name = f"outputs/{date_part}/{time_part}/{filename}"
            else:
                display_name = f"outputs/{rel_path}"
            checkpoint_files.append((display_name, str(checkpoint_file)))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
    
    return checkpoint_files


def get_checkpoint_info(checkpoint_path):
    """
    Get basic information about a checkpoint without fully loading it.
    """
    try:
        # Load only the metadata
        device = 'cpu'  # Load on CPU for inspection
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        info = {
            'step': checkpoint.get('step', 'Unknown'),
            'file_size': os.path.getsize(checkpoint_path) / (1024**2),  # MB
            'modified': datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Try to extract config info
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            info.update({
                'n_layer': getattr(cfg.model, 'n_layer', 'Unknown'),
                'n_head': getattr(cfg.model, 'n_head', 'Unknown'),
                'n_embd': getattr(cfg.model, 'n_embd', 'Unknown'),
                'experiment': getattr(cfg, 'experiment_name', 'Unknown')
            })
        
        return info
    except Exception as e:
        return {'error': str(e)}


@st.cache_resource
def load_model(checkpoint_path):
    """
    Load a trained model from checkpoint.
    """
    try:
        # Load checkpoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get config
        cfg = checkpoint['config']
        
        # Initialize model using Hydra instantiate
        model = instantiate(cfg.model.gpt)
        
        # Load model state
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        return model, cfg, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def generate_text(model, prompt_text, max_tokens, temperature, top_k, device):
    """
    Generate text continuation from a prompt.
    """
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize prompt
    if prompt_text.strip():
        prompt_tokens = torch.tensor([enc.encode(prompt_text)], dtype=torch.long, device=device)
    else:
        prompt_tokens = None
    
    # Create generator for reproducible results
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    # Generate tokens
    with st.spinner("Generating text..."):
        start_time = time.time()
        output_tokens = sample_from_model(
            model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            generator=generator
        )
        generation_time = time.time() - start_time
    
    # Decode output
    full_text = enc.decode(output_tokens[0].cpu().numpy())
    
    if prompt_text.strip():
        generated_text = full_text[len(prompt_text):]
    else:
        generated_text = full_text
    
    return full_text, generated_text, generation_time


def main():
    st.set_page_config(
        page_title="Modded NanoGPT Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Modded NanoGPT Interactive Demo")
    st.markdown("Select a trained checkpoint and interact with your GPT model!")
    
    # Create checkpoints directory if it doesn't exist
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    
    # Sidebar for model loading and configuration
    with st.sidebar:
        st.header("⚙️ Model Selection")
        
        # Find available checkpoints
        checkpoint_files = find_checkpoint_files()
        
        if not checkpoint_files:
            st.warning("🔍 No checkpoint files found!")
            st.markdown("""
            **To add checkpoints:**
            1. Copy your `.pt` files to the `checkpoints/` folder, or
            2. Train models using `train.py` (saves to `outputs/`)
            
            **Current search locations:**
            - `checkpoints/` folder
            - `outputs/` folder (training runs)
            """)
            st.stop()
        
        # Model selection dropdown
        checkpoint_options = [f"{name}" for name, path in checkpoint_files]
        selected_checkpoint = st.selectbox(
            "Select Model Checkpoint:",
            options=checkpoint_options,
            help="Choose from available trained models"
        )
        
        # Get the selected checkpoint path
        selected_path = None
        for name, path in checkpoint_files:
            if name == selected_checkpoint:
                selected_path = path
                break
        
        if selected_path:
            # Show checkpoint info
            st.subheader("📋 Checkpoint Info")
            info = get_checkpoint_info(selected_path)
            
            if 'error' in info:
                st.error(f"Error reading checkpoint: {info['error']}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Step", info.get('step', 'Unknown'))
                    st.metric("File Size", f"{info.get('file_size', 0):.1f} MB")
                with col2:
                    st.metric("Layers", info.get('n_layer', 'Unknown'))
                    st.metric("Heads", info.get('n_head', 'Unknown'))
                
                st.caption(f"**Modified:** {info.get('modified', 'Unknown')}")
                st.caption(f"**Experiment:** {info.get('experiment', 'Unknown')}")
            
            # Load model button
            if st.button("🚀 Load Model", type="primary", use_container_width=True):
                # Clear any previously cached model
                if 'model' in st.session_state:
                    del st.session_state['model']
                load_model.clear()  # Clear streamlit cache
                
                with st.spinner("Loading model..."):
                    model, cfg, device = load_model(selected_path)
                
                if model is not None:
                    st.success("✅ Model loaded successfully!")
                    
                    # Display detailed model info
                    st.subheader("📊 Model Information")
                    st.write(f"**Device:** {device}")
                    st.write(f"**Layers:** {cfg.model.n_layer}")
                    st.write(f"**Heads:** {cfg.model.n_head}")
                    st.write(f"**Embedding Dim:** {cfg.model.n_embd}")
                    st.write(f"**Vocab Size:** {cfg.model.vocab_size}")
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    st.write(f"**Total Parameters:** {total_params:,}")
                    st.write(f"**Trainable Parameters:** {trainable_params:,}")
                    
                    # Store in session state
                    st.session_state['model'] = model
                    st.session_state['cfg'] = cfg
                    st.session_state['device'] = device
                    st.session_state['checkpoint_name'] = selected_checkpoint
                    
                    st.rerun()  # Refresh to show generation interface
        
        # Generation parameters (only show if model is loaded)
        if 'model' in st.session_state:
            st.subheader("🎛️ Generation Settings")
            max_tokens = st.slider("Max New Tokens", 1, 512, 128, help="Maximum number of tokens to generate")
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1, help="Controls randomness in generation")
            top_k = st.slider("Top-k", 1, 100, 50, help="Number of top tokens to sample from")
            
            # Store in session state
            st.session_state['max_tokens'] = max_tokens
            st.session_state['temperature'] = temperature
            st.session_state['top_k'] = top_k
    
    # Main content area
    if 'model' in st.session_state:
        model = st.session_state['model']
        cfg = st.session_state['cfg']
        device = st.session_state['device']
        max_tokens = st.session_state['max_tokens']
        temperature = st.session_state['temperature']
        top_k = st.session_state['top_k']
        checkpoint_name = st.session_state['checkpoint_name']
        
        # Show currently loaded model
        st.info(f"🎯 **Currently loaded:** {checkpoint_name}")
        
        # Text generation interface
        st.header("💬 Text Generation")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Input")
            
            # Handle example prompt selection
            default_value = ""
            if 'example_prompt' in st.session_state:
                default_value = st.session_state['example_prompt']
                # Clear the example prompt after using it
                del st.session_state['example_prompt']
            
            prompt_text = st.text_area(
                "Enter your prompt:",
                value=default_value,
                height=200,
                placeholder="Type your prompt here, or leave empty for unconditional generation...",
                help="Enter text to continue, or leave empty to generate from scratch"
            )
            
            generate_button = st.button("🚀 Generate Text", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("🎯 Output")
            if generate_button and prompt_text is not None:
                try:
                    full_text, generated_text, gen_time = generate_text(
                        model, prompt_text, max_tokens, temperature, top_k, device
                    )
                    
                    # Display results
                    if prompt_text.strip():
                        st.markdown("**Full Text (Prompt + Generated):**")
                        st.text_area("", value=full_text, height=200, disabled=True)
                        st.markdown("**Generated Continuation:**")
                        st.text_area("", value=generated_text, height=100, disabled=True)
                    else:
                        st.markdown("**Generated Text:**")
                        st.text_area("", value=generated_text, height=200, disabled=True)
                    
                    # Show generation stats
                    tokens_generated = len(generated_text.split())
                    tokens_per_second = tokens_generated / gen_time if gen_time > 0 else 0
                    st.caption(f"⏱️ Generated in {gen_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
                    
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
            elif generate_button:
                st.warning("Please enter a prompt or leave empty for unconditional generation")
        
        # Pre-defined examples
        st.header("💡 Example Prompts")
        st.markdown("Click on any example to use it as a prompt:")
        
        example_prompts = [
            "Once upon a time in a distant galaxy,",
            "The future of artificial intelligence",
            "In the year 2050, technology will",
            "The secret to happiness is",
            "Climate change is a global challenge that",
            "The most important lesson I learned was",
            "In a world where robots and humans coexist,",
            "The discovery of the ancient artifact"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(example_prompts):
            with cols[i % 2]:
                if st.button(f"📌 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state['example_prompt'] = example
                    st.rerun()

    
    else:
        # Instructions when no model is loaded
        st.header("🚀 Getting Started")
        st.markdown(f"""
        To use this demo:
        
        1. **Add checkpoints** to the `checkpoints/` folder or train models using `train.py`
        2. **Select a model** from the dropdown in the sidebar
        3. **Load the model** by clicking the "Load Model" button
        4. **Adjust generation parameters** (temperature, top-k, etc.)
        5. **Enter a prompt** or leave it empty for unconditional generation
        6. **Click Generate** to see the model's output!
        
        ### 📁 Checkpoint Locations
        The demo automatically searches for `.pt` files in:
        - **`{CHECKPOINTS_DIR}/`** - Dedicated checkpoint folder (recommended)
        - **`{OUTPUTS_DIR}/`** - Training output folder (from `train.py`)
        
        ### 📚 About the Models
        This demo supports models from the Modded NanoGPT project, which implements:
        - **GPT architecture** with custom mixing layers
        - **Polynomial Mixer (PoM)** attention mechanism  
        - **SOAP optimizer** for improved training
        - **Multiple model sizes** and configurations
        
        ### 🎯 Checkpoint Format
        Checkpoint files should contain:
        - `model`: Model state dictionary
        - `config`: Hydra configuration object
        - `optimizer`: Optimizer state (optional for inference)
        - `step`: Training step number
        """)


if __name__ == "__main__":
    # Set up tiktoken cache
    os.environ["TIKTOKEN_CACHE_DIR"] = ".tiktoken_cache"
    Path(".tiktoken_cache").mkdir(parents=True, exist_ok=True)
    
    main() 