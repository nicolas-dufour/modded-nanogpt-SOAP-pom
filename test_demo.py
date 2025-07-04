#!/usr/bin/env python3
"""
Simple test script to validate the Streamlit demo setup.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    
    try:
        import tiktoken
        print("✅ tiktoken imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tiktoken: {e}")
        return False
    
    try:
        from models.gpt import GPT
        print("✅ GPT model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GPT model: {e}")
        return False
    
    try:
        import models.pom as pom
        print("✅ PoM module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PoM module: {e}")
        return False
    
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        print("✅ Hydra and OmegaConf imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Hydra/OmegaConf: {e}")
        return False
    
    return True


def test_tiktoken_setup():
    """Test tiktoken setup and caching."""
    print("\n🔍 Testing tiktoken setup...")
    
    try:
        # Set up cache directory
        os.environ["TIKTOKEN_CACHE_DIR"] = ".tiktoken_cache"
        Path(".tiktoken_cache").mkdir(parents=True, exist_ok=True)
        
        # Test tokenizer
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        
        # Test encoding/decoding
        test_text = "Hello, world!"
        tokens = enc.encode(test_text)
        decoded = enc.decode(tokens)
        
        assert decoded == test_text, f"Encoding/decoding mismatch: {decoded} != {test_text}"
        print(f"✅ tiktoken working correctly (encoded {len(tokens)} tokens)")
        return True
        
    except Exception as e:
        print(f"❌ tiktoken test failed: {e}")
        return False


def test_model_creation():
    """Test basic model creation."""
    print("\n🔍 Testing model creation...")
    
    try:
        from models.gpt import GPT, CausalSelfPoM
        
        # Create a simple PoM layer
        pom_layer = CausalSelfPoM(n_embd=64, degree=2, expand=2, n_head=1)
        print("✅ PoM layer created successfully")
        
        # Create a minimal GPT model
        model = GPT(
            mixing_layer=pom_layer,
            vocab_size=1000,  # Small vocab for testing
            n_layer=2,        # Small model for testing
            n_head=1,
            n_embd=64
        )
        print(f"✅ GPT model created successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        # Test forward pass
        import torch
        x = torch.randint(0, 1000, (1, 10))  # Batch=1, seq_len=10
        with torch.no_grad():
            logits, loss = model(x, targets=None, return_logits=True)
        
        assert logits.shape == (1, 1, 1000), f"Unexpected logits shape: {logits.shape}"
        print("✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_checkpoint_discovery():
    """Test checkpoint discovery functionality."""
    print("\n🔍 Testing checkpoint discovery...")
    
    try:
        # Import the demo module to test checkpoint finding
        sys.path.append(str(project_root))
        from streamlit_demo import find_checkpoint_files, CHECKPOINTS_DIR, OUTPUTS_DIR
        
        # Test that directories are created
        CHECKPOINTS_DIR.mkdir(exist_ok=True)
        print(f"✅ Checkpoints directory: {CHECKPOINTS_DIR}")
        
        # Test checkpoint discovery
        checkpoint_files = find_checkpoint_files()
        print(f"✅ Checkpoint discovery working (found {len(checkpoint_files)} files)")
        
        if checkpoint_files:
            print("📋 Available checkpoints:")
            for name, path in checkpoint_files[:3]:  # Show first 3
                print(f"  - {name}")
            if len(checkpoint_files) > 3:
                print(f"  ... and {len(checkpoint_files) - 3} more")
        else:
            print("ℹ️  No checkpoint files found (this is normal for a fresh setup)")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint discovery test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Modded NanoGPT Demo Setup")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("tiktoken Test", test_tiktoken_setup),
        ("Model Creation Test", test_model_creation),
        ("Checkpoint Discovery Test", test_checkpoint_discovery),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The demo should work correctly.")
        print("\n💡 To run the demo:")
        print("   ./run_demo.sh")
        print("   OR")
        print("   streamlit run streamlit_demo.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 