# File: penguins_ai/train/train_nhl_optimized.py
# Add these imports at the top
import sys
sys.path.append('.')
from config.optimal_settings import SYSTEM_OPTIMAL_CONFIG

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time

class OptimizedNHLTrainer:
    """NHL model training optimized for your P16 system"""
    
    def __init__(self):
        self.config = SYSTEM_OPTIMAL_CONFIG
        self.xgb_params = self.config['xgboost_config'].copy()
        
        # ADD GPU CONFIGURATION
        if self.config.get('use_gpu', False):
            self.xgb_params['tree_method'] = 'gpu_hist'
            self.xgb_params['predictor'] = 'gpu_predictor'
            self.xgb_params['gpu_id'] = 0
            print("🎮 GPU acceleration enabled for XGBoost")
        
        print(f"🖥️ System Configuration Loaded:")
        print(f"  - XGBoost threads: {self.xgb_params['n_jobs']}")
        print(f"  - Max memory: {self.config['max_memory_gb']}GB")
        print(f"  - Batch size: {self.config['batch_size']}")
        print(f"  - GPU enabled: {self.config['use_gpu']}")
        print(f"  - Tree method: {self.xgb_params.get('tree_method', 'auto')}")
    
    # ... rest of your methods stay the same ...
    
    def calculate_traffic(self, shot, previous_plays):
        """Estimate traffic in front of net"""
        # This method needs to be inside the class
        return len([p for p in previous_plays[-3:] if 'hit' in p.get('typeDescKey', '')])
    
    def train_xgboost_model(self, df):
        """Train model with optimal settings"""
        # This entire method must be indented to be inside the class
        print(f"\n🤖 Training XGBoost with {self.xgb_params['n_jobs']} threads...")
        
        # Show GPU status
        if self.xgb_params.get('tree_method') == 'gpu_hist':
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
                    print(f"  📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            except:
                pass
        
        # Rest of the method continues here...
        # Make sure it's all indented properly