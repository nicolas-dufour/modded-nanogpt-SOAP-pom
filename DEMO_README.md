# 🤖 Modded NanoGPT Interactive Demo

A Streamlit-based web interface for interacting with your trained GPT models from the Modded NanoGPT project. **Key feature: Easily switch between different trained models to compare their outputs!**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Organize Your Checkpoints (Optional)

If you have trained models in the `outputs/` directory, you can organize them:

```bash
# Preview what would be organized
python organize_checkpoints.py --organize

# Actually organize the files
python organize_checkpoints.py --organize --execute

# List organized checkpoints
python organize_checkpoints.py --list
```

### 3. Launch the Demo

**Option A: Using the launcher script**
```bash
./run_demo.sh
```

**Option B: Direct Streamlit command**
```bash
streamlit run streamlit_demo.py
```

### 3. Open Your Browser

The demo will be available at: http://localhost:8501

## 📋 How to Use

1. **Add Checkpoints**: Copy your trained `.pt` files to the `checkpoints/` folder (or use existing ones from `outputs/`)
2. **Select a Model**: Choose from the dropdown list of available checkpoints in the sidebar
3. **Load the Model**: Click the "Load Model" button to initialize the selected checkpoint
4. **Configure Generation**: Adjust parameters like temperature, top-k, and max tokens  
5. **Enter a Prompt**: Type your text prompt or leave empty for unconditional generation
6. **Generate**: Click the "Generate Text" button to see your model's output
7. **Switch Models**: Select a different checkpoint and reload to compare models

## 🎛️ Generation Parameters

- **Max New Tokens**: Maximum number of tokens to generate (1-512)
- **Temperature**: Controls randomness (0.1 = deterministic, 2.0 = very random)
- **Top-k**: Number of top tokens to sample from (1-100)

## 📊 Model Information

The demo displays:
- Model architecture details (layers, heads, embedding dimensions)
- Parameter counts (total and trainable)
- Device information (CPU/GPU)
- Generation speed metrics

## 📁 Checkpoint Organization

The demo automatically searches for `.pt` files in two locations:

### 1. Dedicated Checkpoints Folder
```
checkpoints/
├── experiment1/
│   ├── model_step_1000.pt
│   └── model_step_2000.pt
├── experiment2/
│   └── final_model.pt
└── best_models/
    ├── pom_baseline.pt
    └── transformer_baseline.pt
```

### 2. Training Outputs (Automatic)
The demo also finds checkpoints saved during training:
```
outputs/
├── 2025-07-03/
│   ├── 15-52-44/
│   │   └── state_step001500.pt
│   └── 14-35-42/
│       └── state_step001000.pt
└── 2025-07-01/
    └── 16-30-47/
        └── state_step002000.pt
```

**Tips for Organization:**
- Use descriptive folder names (e.g., `pom_experiments/`, `baselines/`)
- Include training step in filename for easy identification
- Group related experiments together
- The demo shows file modification time and step count for easy comparison

## 🎯 Checkpoint Requirements

Your checkpoint file must contain:
- `model`: Model state dictionary
- `config`: Hydra configuration object 
- `optimizer`: Optimizer state (optional for inference)
- `step`: Training step number

This format is automatically created when training with `train.py`.

## 🔄 Model Switching & Comparison

One of the key features of this demo is the ability to easily switch between different trained models:

1. **Quick Selection**: Use the dropdown to browse all available checkpoints
2. **Model Preview**: See checkpoint info (step, size, architecture) before loading
3. **Easy Switching**: Load a different model with one click
4. **Comparison Workflow**: 
   - Generate text with Model A
   - Switch to Model B
   - Use the same prompt to compare outputs
   - Experiment with different architectures (PoM vs Self-Attention)

**Model Information Display:**
- Training step and experiment name
- Model architecture (layers, heads, embedding size)
- File size and modification time
- Parameter count (total and trainable)

## 💡 Example Prompts

The demo includes several pre-defined example prompts:
- "Once upon a time in a distant galaxy,"
- "The future of artificial intelligence"
- "In the year 2050, technology will"
- "The secret to happiness is"
- And more...

## 🔧 Troubleshooting

### Model Loading Issues

If you encounter errors loading your model:
1. Ensure the checkpoint was saved correctly during training
2. Check that all required model files are present in the project
3. Verify the checkpoint contains the required keys (`model`, `config`)

### Memory Issues

For large models:
- Use a machine with sufficient GPU memory
- Consider reducing `max_new_tokens` if generation is slow
- The demo automatically handles GPU memory cleanup

### Dependencies

If you get import errors:
```bash
pip install --upgrade -r requirements.txt
```

## 🏗️ Architecture Support

The demo supports models with:
- **Standard Self-Attention** (`CausalSelfAttention`)
- **Polynomial Mixer (PoM)** (`CausalSelfPoM`)
- **Various model sizes** (d12, d24, d36, d48)
- **Different optimizers** (AdamW, SOAP)

## 🚀 Advanced Usage

### Custom Model Configurations

The demo automatically loads model configurations from the checkpoint. If you need to modify the configuration:

1. Load the checkpoint manually
2. Modify the `cfg` object
3. Reinstantiate the model with the new config

### Batch Generation

Currently, the demo generates one sample at a time. For batch generation, you can modify the `generate_text()` function to process multiple prompts simultaneously.

### Custom Sampling

The sampling parameters can be extended by modifying the `sample_from_model()` function to support:
- Nucleus (top-p) sampling
- Repetition penalties
- Custom stopping criteria

## 📝 Development

To modify or extend the demo:

1. **Add new parameters**: Update the sidebar sliders in `main()`
2. **Custom generation**: Modify `sample_from_model()` function
3. **UI improvements**: Update the Streamlit layout and components
4. **Model support**: Extend `load_model()` for different architectures

## 🤝 Contributing

Feel free to submit issues and feature requests! The demo is designed to be easily extensible for new model architectures and generation techniques. 