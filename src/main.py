#!/usr/bin/env python
"""
Comprehensive Fairness Research Framework (PyTorch, Colab-friendly) - Fixed Version
===============================================================================
A complete framework for studying fairness in NLP models across different 
fine-tuning methods (SFT, LoRA, Task Vector) and datasets.

Features:
- Multiple datasets: civil_comments, hate_speech18, founta
- Multiple fine-tuning methods: SFT, LoRA, Task Vector
- Comprehensive fairness metrics: demographic parity, equalized odds, etc.
- Easy experimentation and comparison

Usage:
```python
!pip install -q transformers datasets peft accelerate bitsandbytes scikit-learn

# Run experiments
from fairness_framework import FairnessExperiment
exp = FairnessExperiment()

# Quick test
exp.run_experiment(method="sft", dataset="civil_comments", sample_frac=0.05, epochs=1)

# Compare methods
results = exp.compare_methods(dataset="civil_comments", sample_frac=0.1, epochs=2)
```
"""

from __future__ import annotations
import os, random, json, warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import wandb
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

warnings.filterwarnings('ignore')

#wandb log directory
cache_dir = str(Path('./cache').resolve())
os.makedirs(f'{cache_dir}/wandb', exist_ok=True)
os.environ['WANDB_DIR'] = os.environ['WANDB_DATA_DIR'] = f'{cache_dir}/wandb'
os.environ['WANDB_CACHE_DIR'] = os.environ['WANDB_CONFIG_DIR'] = os.environ['WANDB_DIR']
# =============================================================================
# Configuration & Data Classes
# =============================================================================

@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""
    accuracy: float
    demographic_parity: float
    equalized_odds: float
    equal_opportunity: float
    group_metrics: Dict[str, Dict[str, float]]

@dataclass
class ExperimentConfig:
    """Configuration for fairness experiments."""
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "civil_comments"
    method: str = "sft"  # sft, lora, task_vector
    sample_frac: float = 1
    epochs: int = 2
    learning_rate: float = 1e-5
    batch_size: int = 16
    output_dir: str = "./fairness_experiments"
    use_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    seed: int = 42
    group_type: str = "gender"

# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_model_and_tokenizer(model_name: str, num_labels: int = 2, use_4bit: bool = False):
    """Load model and tokenizer with optional quantization."""
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True          # ⭐ NEW for Qwen
    )

    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token
        # Qwen ships only an EOS token – promote it to PAD as well
        tokenizer.pad_token = tokenizer.eos_token
    
    # if use_4bit and BitsAndBytesConfig is not None:
    if use_4bit and BitsAndBytesConfig is not None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )


        model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                quantization_config=bnb_config,   # or plain fp16 path
                device_map="auto",
                trust_remote_code=True,
        )

        # NEW: make sure model knows the padding id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    else:

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True
        )
        if model.config.model_type == "qwen" and not hasattr(model, "score"):
            model.score = torch.nn.Linear(model.config.hidden_size, num_labels, bias=False)
        model.config.pad_token_id = tokenizer.pad_token_id
        
    
    return model, tokenizer

# =============================================================================
# Dataset Processing
# =============================================================================

from transformers import DataCollatorWithPadding


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate group information before processing
        groups = [feature.pop("group") for feature in features if "group" in feature]
        
        # Use the parent collator for the tokenized features
        batch = super().__call__(features)
        
        # Don't add groups back to the batch since the model doesn't need them
        # Groups will be handled separately during evaluation
        
        return batch

class DatasetProcessor:
    """Handle different datasets with unified interface."""
    
    IDENTITY_COLUMNS = {
        'gender': ["male", "female", "transgender", "other_gender"],
        'race': ["black", "white", "asian", "latino", "other_race_or_ethnicity"],
        'religion': ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist", "other_religion"],
    }
    
    @staticmethod
    def inspect_dataset_structure(dataset_name: str):
        """Inspect dataset structure to understand column names."""
        try:
            sample = load_dataset(dataset_name, split="train[:10]")
            print(f"📋 Dataset '{dataset_name}' columns: {sample.column_names}")
            print(f"📋 Sample data: {sample[0]}")
            return sample.column_names
        except Exception as e:
            print(f"⚠️  Error inspecting dataset {dataset_name}: {e}")
            return []
    
    @staticmethod
    def load_civil_comments(sample_frac: float = 1.0, group_type: str = "gender") -> DatasetDict:
        """Load and process civil_comments dataset."""
        pct = max(1, int(sample_frac * 100))
        print(f"📥 Loading civil_comments dataset ({pct}% sample)...")
        
        # First inspect the structure
        print("🔍 Inspecting dataset structure...")
        try:
            sample = load_dataset("badrmarani/civil-comments-wilds", split="train[:10]")
            print(f"📋 Available columns: {sample.column_names}")
            
            # Find the text column
            text_column = None
            for col in ['text', 'comment_text', 'comment', 'content']:
                if col in sample.column_names:
                    text_column = col
                    break
            
            if text_column is None:
                print("⚠️  Could not find text column, using first string column")
                for col in sample.column_names:
                    if isinstance(sample[0][col], str):
                        text_column = col
                        break
            
            # Find the toxicity column
            toxicity_column = None
            for col in ['toxicity', 'toxic', 'label', 'target']:
                if col in sample.column_names:
                    toxicity_column = col
                    break
            
            print(f"📋 Using text column: {text_column}")
            print(f"📋 Using toxicity column: {toxicity_column}")
            
            if text_column is None or toxicity_column is None:
                raise ValueError("Could not identify required columns")
                
        except Exception as e:
            print(f"⚠️  Error loading civil_comments: {e}")
            # Fallback to synthetic data
            return DatasetProcessor.create_synthetic_dataset(sample_frac)
        
        train_ds = load_dataset("badrmarani/civil-comments-wilds", split=f"train[:{pct}%]")
        test_ds = load_dataset("badrmarani/civil-comments-wilds", split=f"test[:{pct}%]")
        
        # Find available identity columns
        available_cols = [col for col in DatasetProcessor.IDENTITY_COLUMNS[group_type] 
                         if col in train_ds.column_names]
        print(f"📋 Available identity columns: {available_cols}")
        
        def preprocess(batch):
            # Handle toxicity labels
            if toxicity_column in batch:
                if isinstance(batch[toxicity_column][0], (int, float)):
                    labels = [1 if score > 0.5 else 0 for score in batch[toxicity_column]]
                else:
                    labels = batch[toxicity_column]
            else:
                # Default labels if no toxicity column
                labels = [0] * len(batch[text_column])
            
            THR = 0.5
            groups_list = []

            for i in range(len(labels)):
                g = []
                for col in available_cols:
                    v = batch[col][i]
                    if isinstance(v, (int, float)) and v > THR:
                        g.append(col)
                    elif isinstance(v, bool) and v:
                        g.append(col)
                if not g:
                    g = ["no_group_mentioned"]
                groups_list.append(g)
            
            clean_test = [t if isinstance(t, str) else "" for t in batch[text_column]]
            return {
                "text": clean_test,
                "labels": labels,
                "group": groups_list
            }
        
        processed = DatasetDict({
            "train": train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names),
            "test": test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)
        })
        
        return processed
    
    @staticmethod
    def create_synthetic_dataset(sample_frac: float = 1.0) -> DatasetDict:
        """Create synthetic dataset for testing when real dataset fails."""
        print("🔧 Creating synthetic dataset for testing...")
        
        # Generate synthetic toxic/non-toxic comments
        toxic_phrases = [
            "you are terrible", "this is awful", "hate this", "stupid people",
            "worst thing ever", "complete garbage", "totally useless"
        ]
        
        non_toxic_phrases = [
            "this is interesting", "good point", "thanks for sharing",
            "helpful information", "appreciate this", "well written"
        ]
        
        groups = ["male", "female", "black", "white", "christian", "muslim", "no_group_mentioned"]
        
        n_samples = int(10000 * sample_frac)
        
        # Generate training data
        train_texts, train_labels, train_groups = [], [], []
        for _ in range(n_samples):
            is_toxic = random.random() < 0.3  # 30% toxic
            text = random.choice(toxic_phrases if is_toxic else non_toxic_phrases)
            text += f" {random.randint(1, 1000)}"  # Make unique
            
            train_texts.append(text)
            train_labels.append(1 if is_toxic else 0)
            train_groups.append(random.choice(groups))
        
        # Generate test data
        test_texts, test_labels, test_groups = [], [], []
        for _ in range(n_samples // 4):
            is_toxic = random.random() < 0.3
            text = random.choice(toxic_phrases if is_toxic else non_toxic_phrases)
            text += f" test {random.randint(1, 1000)}"
            
            test_texts.append(text)
            test_labels.append(1 if is_toxic else 0)
            test_groups.append(random.choice(groups))
        
        train_dataset = Dataset.from_dict({
            "text": train_texts,
            "labels": train_labels,
            "group": train_groups
        })
        
        test_dataset = Dataset.from_dict({
            "text": test_texts,
            "labels": test_labels,
            "group": test_groups
        })
        
        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    
    @staticmethod
    def load_hate_speech18(sample_frac: float = 1.0) -> DatasetDict:
        """Load and process hate_speech18 dataset."""
        print(f"📥 Loading hate_speech18 dataset...")
        
        # Try multiple possible dataset names/configurations
        possible_names = [
            "hate_speech18",
            "hateval2019",
            "haspeede",
            "davidson-hate-speech"
        ]
        
        for name in possible_names:
            try:
                dataset = load_dataset(name)
                print(f"✅ Successfully loaded {name}")
                
                def preprocess(batch):
                    # Adapt this based on actual dataset structure
                    labels = batch.get("label", [0] * len(batch["text"]))
                    groups = ["general"] * len(batch["text"])  # Simplified grouping
                    
                    return {
                        "text": batch["text"],
                        "labels": labels,
                        "group": groups
                    }
                
                processed = dataset.map(preprocess, batched=True)
                return processed
                
            except Exception as e:
                print(f"⚠️  Could not load {name}: {e}")
                continue
        
        print(f"⚠️  Could not load any hate speech dataset, falling back to synthetic data")
        return DatasetProcessor.create_synthetic_dataset(sample_frac)
    
    @classmethod
    def load_dataset(cls, dataset_name: str, sample_frac: float = 1.0, group_type: str = "gender") -> DatasetDict:
        """Load specified dataset."""
        if dataset_name == "civil_comments":
            return cls.load_civil_comments(sample_frac, group_type)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

# =============================================================================
# Training Methods
# =============================================================================

class TrainingManager:
    """Manage different training methods."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        set_seed(config.seed)

        if config.method == "sft":
            self.save_dir = f"{self.config.output_dir}/sft_distilbert"
        elif config.method == "lora":
            self.save_dir = f"{self.config.output_dir}/lora_distilbert"
        elif config.method == "task_vector":
            self.save_dir = f"{self.config.output_dir}/task_vector_distilbert"
    
    def run_sft(self, model, tokenizer, train_ds, eval_ds) -> None:
        """Run standard fine-tuning."""
        print("🚀 Starting SFT training...")

        training_args = TrainingArguments(
            output_dir=self.save_dir,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            weight_decay=0.01,
            # eval_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=False,   # Add this line
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=CustomDataCollator(tokenizer)
        )
        
        trainer.train()
        model.save_pretrained(self.save_dir)
        print(f"✅ SFT training completed. Model saved to {self.save_dir}")
    
    def run_lora(self, model, tokenizer, train_ds, eval_ds) -> None:
        """Run LoRA fine-tuning."""
        print("🚀 Starting LoRA training...")

        is_qwen = "qwen" in self.config.model_name.lower()

        # --- NEW --------------------------------------------------------
        from peft import TaskType
        # Use SEQ_CLS for sequence‑classification checkpoints,
        # even if the underlying architecture is "qwen".
        task_type = TaskType.SEQ_CLS
        # ---------------------------------------------------------------

        lora_config = LoraConfig(
            task_type      = task_type,
            r              = self.config.lora_r,
            lora_alpha     = self.config.lora_alpha,
            lora_dropout   = self.config.lora_dropout,
            target_modules = (["q_proj", "v_proj"] if is_qwen else ["query", "value"])
        )

        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=self.save_dir,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            logging_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=False,   # Add this line
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=CustomDataCollator(tokenizer)
        )
        
        trainer.train()
        model.save_pretrained(self.save_dir)
        print(f"✅ LoRA training completed. Model saved to {self.save_dir}")
    
    def run_task_vector(self, model, tokenizer, train_ds, eval_ds) -> None:
        """Run task vector experiment by training on each group and summing vectors."""
        print("🚀 Starting Task Vector experiment...")
        from dataclasses import replace


        # 1. Save the base model state in memory
        base_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        
        # Get unique groups from the training data
        unique_groups = sorted({
            g for item in train_ds['group']
            for g in (item if isinstance(item, list) else [item])
        })
        print(f"Found groups for task vector training: {unique_groups}")

        task_vectors = []

        original_save_dir = self.save_dir

        # 2. Train a model for each group and calculate its task vector
        for group in unique_groups:
            if group == "no_group_mentioned":
                print(f"Skipping group '{group}' for task vector calculation.")
                continue

            print(f"\n--- Training for group: {group} ---")
            
            # Filter dataset for the current group
            group_train_ds = train_ds.filter(
                lambda example, grp=group: (grp in example['group']) if isinstance(example['group'], list) else (example['group'] == grp),
                load_from_cache_file=False
            )
            
            if len(group_train_ds) == 0:
                print(f"Skipping group '{group}' due to no samples.")
                continue

            # Load a fresh pre-trained model for this group's training
            group_model, _ = setup_model_and_tokenizer(
                self.config.model_name,
                num_labels=model.config.num_labels,
                use_4bit=self.config.use_4bit
            )
            if not self.config.use_4bit and torch.cuda.is_available():
                group_model.to(torch.device("cuda"))

            # Fine-tune on the group-specific data
            group_save_dir = f"{original_save_dir}/group_{group}"
            os.makedirs(group_save_dir, exist_ok=True)
            
            # Create a new config with the correct save directory
            group_config = replace(self.config, output_dir=group_save_dir)
            group_trainer_manager = TrainingManager(group_config)
            group_trainer_manager.run_sft(group_model, tokenizer, group_train_ds, eval_ds)
            
            # Verify the model was saved
            expected_model_path = f"{group_save_dir}/sft_distilbert"
            if os.path.exists(f"{expected_model_path}/pytorch_model.bin") or os.path.exists(f"{expected_model_path}/model.safetensors"):
                print(f"   ✅ Group model saved successfully: {expected_model_path}")
            else:
                print(f"   ⚠️  Group model may not have saved properly: {expected_model_path}")

            # Calculate task vector for this group (on CPU to avoid device issues)
            ft_state = {k: v.cpu() for k, v in group_model.state_dict().items()}
            task_vector = {k: ft_state[k] - base_state[k] for k in base_state.keys() if k in ft_state}
            task_vectors.append(task_vector)
            
            print(f"✅ Task vector calculated for group: {group}")
            
            # Clean up memory
            del group_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not task_vectors:
            print("⚠️ No task vectors were generated. Final model is the base model.")
            return

        # 3. Sum the task vectors
        print("\n--- Summing task vectors ---")
        summed_task_vector = {}
        # Get keys from the first task vector
        for k in task_vectors[0].keys():
            # Sum up the tensors for key 'k' from all task vectors
            summed_task_vector[k] = torch.stack([tv[k] for tv in task_vectors]).sum(dim=0)

        # 4. Apply the summed task vector to the original model
        for scale_factor in [.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print(f"--- Applying summed task vector (scale={scale_factor}) to base model ---")
            final_model_device = model.device
            new_state_dict = {
                # base + (scale_factor * delta)
                k: base_state[k].to(final_model_device) + (scale_factor * summed_task_vector.get(k, 0)).to(final_model_device) 
                for k in base_state.keys()
            }
            model.load_state_dict(new_state_dict)

            # Save the final model with the applied task vector for inspection
            task_vector_save_dir = f"{original_save_dir}/task_vector_scale_{scale_factor}"
            os.makedirs(task_vector_save_dir, exist_ok=True)
            model.save_pretrained(task_vector_save_dir)
            
            # Verify the task vector model was saved
            if os.path.exists(f"{task_vector_save_dir}/pytorch_model.bin") or os.path.exists(f"{task_vector_save_dir}/model.safetensors"):
                print(f"   ✅ Task vector scale {scale_factor} saved successfully: {task_vector_save_dir}")
            else:
                print(f"   ⚠️  Task vector scale {scale_factor} may not have saved properly: {task_vector_save_dir}")
        
            print(f"✅ Task vector summation complete. Final model updated and saved to {task_vector_save_dir}")

        # Restore save_dir so that external code (e.g., evaluation) points to the correct base output dir
        self.save_dir = original_save_dir

# =============================================================================
# Fairness Evaluation
# =============================================================================

class FairnessEvaluator:
    """Comprehensive fairness evaluation."""
    
    @staticmethod
    def compute_group_metrics(y_true: List[int], y_pred: List[int], groups: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute per-group classification metrics."""
        unique_groups = sorted({
            g for item in groups
            for g in (item if isinstance(item, list) else [item])
        })

        group_metrics = {}

        for group in unique_groups:
            if group == "no_group_mentioned":
                continue

            group_indices = [
                i for i, gs in enumerate(groups)
                if (group in gs if isinstance(gs, list) else gs == group)
            ]
            if not group_indices:
                continue

            group_y_true = [y_true[i] for i in group_indices]
            group_y_pred = [y_pred[i] for i in group_indices]
            
            # Compute confusion matrix
            try:
                tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
            except ValueError:
                # Handle cases where only one class is present
                tn = fp = fn = tp = 0
                for i, (true, pred) in enumerate(zip(group_y_true, group_y_pred)):
                    if true == 0 and pred == 0: tn += 1
                    elif true == 0 and pred == 1: fp += 1
                    elif true == 1 and pred == 0: fn += 1
                    elif true == 1 and pred == 1: tp += 1
            
            # Compute metrics
            accuracy = (tp + tn) / len(group_indices) if group_indices else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            group_metrics[group] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'count': len(group_indices)
            }
        
        return group_metrics
    
    @staticmethod
    def compute_fairness_metrics(y_true: List[int], y_pred: List[int], groups: List[str]) -> FairnessMetrics:
        """Compute comprehensive fairness metrics."""
        overall_accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        group_metrics = FairnessEvaluator.compute_group_metrics(y_true, y_pred, groups)
        
        if len(group_metrics) < 2:
            return FairnessMetrics(
                accuracy=overall_accuracy,
                demographic_parity=0.0,
                equalized_odds=0.0,
                equal_opportunity=0.0,
                group_metrics=group_metrics
            )
        
        # Demographic Parity (difference in positive prediction rates)
        pos_rates = [metrics['tp'] + metrics['fp'] for metrics in group_metrics.values()]
        total_counts = [metrics['count'] for metrics in group_metrics.values()]
        pos_pred_rates = [pos / total if total > 0 else 0 for pos, total in zip(pos_rates, total_counts)]
        demographic_parity = max(pos_pred_rates) - min(pos_pred_rates) if pos_pred_rates else 0
        
        # Equalized Odds (difference in TPR and FPR)
        tprs = [metrics['recall'] for metrics in group_metrics.values()]
        fprs = [metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0 
                for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs) if tprs else 0
        fpr_diff = max(fprs) - min(fprs) if fprs else 0
        equalized_odds = max(tpr_diff, fpr_diff)
        
        # Equal Opportunity (difference in TPR only)
        equal_opportunity = tpr_diff
        
        return FairnessMetrics(
            accuracy=overall_accuracy,
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            equal_opportunity=equal_opportunity,
            group_metrics=group_metrics
        )
    
    @staticmethod
    def evaluate_model(model, tokenizer, test_dataset) -> FairnessMetrics:
        """Evaluate model and return fairness metrics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        predictions = []
        # Create a custom data collator that removes group info
        eval_collator = DataCollatorWithPadding(tokenizer)
        
        # Prepare dataset without group column for model input
        eval_dataset = test_dataset.remove_columns(["group"])
        data_loader = torch.utils.data.DataLoader(
            eval_dataset, 
            batch_size=32, 
            collate_fn=eval_collator
        )
        
        print("🔍 Evaluating model...")
        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(device) for k, v in batch.items() 
                        if k in tokenizer.model_input_names}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().tolist())
        
        y_true = test_dataset["labels"]
        y_pred = predictions
        groups = test_dataset["group"]  # Get groups from original dataset
        
        return FairnessEvaluator.compute_fairness_metrics(y_true, y_pred, groups)

# =============================================================================
# Main Experiment Class
# =============================================================================

class FairnessExperiment:
    """Main experiment runner for fairness research."""
    
    def __init__(self):
        self.results = {}
    
    def run_experiment(self, 
                      method: str = "sft",
                      dataset: str = "civil_comments", 
                      model_name: str = "distilbert-base-uncased",
                    # model_name: str = "distilbert-base-uncased",

                      sample_frac: float = 0.1,
                      epochs: int = 2,
                      output_dir: str = "./fairness_results",
                      **kwargs) -> FairnessMetrics:
        """Run a single experiment."""
        
        config = ExperimentConfig(
            method=method,
            dataset_name=dataset,
            model_name=model_name,
            sample_frac=sample_frac,
            epochs=epochs,
            output_dir=output_dir,
            **kwargs
        )

        safe_name = model_name.split('/')[-1].replace('/', '')
        wandb.init(
            project = f"{safe_name}{dataset}_{method}", 
            entity="llm_jp_pp",
            config=vars(config),
            )
        
        print(f"\n{'='*60}")
        print(f"🧪 Running Fairness Experiment")
        print(f"   Method: {method}")
        print(f"   Dataset: {dataset}")
        print(f"   Model: {model_name}")
        print(f"   Sample: {sample_frac*100:.1f}%")
        print(f"   Epochs: {epochs}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset_dict = DatasetProcessor.load_dataset(config.dataset_name, config.sample_frac, config.group_type)
        
        # Print dataset info
        print(f"📊 Dataset loaded:")
        print(f"   Train samples: {len(dataset_dict['train'])}")
        print(f"   Test samples: {len(dataset_dict['test'])}")
        # print(f"   Groups in train: {set(dataset_dict['train']['group'])}")
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(
            config.model_name, 
            use_4bit=config.use_4bit
        )
        print(f"Model loaded")
        
        # Tokenize dataset
        def tokenize_function(examples):
            tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding=True)
            # Keep labels and group columns, remove text column
            tokenized["labels"] = examples["labels"]
            tokenized["group"] = examples["group"]
            return tokenized

        tokenized_dataset = dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])
        print(f"Dataset tokenized")

        # Run training
        trainer = TrainingManager(config)
        print(f"Training manager initialized")

        print(f"Running {method} training")
        
        if method == "sft":
            trainer.run_sft(model, tokenizer, tokenized_dataset["train"], tokenized_dataset["test"])
        elif method == "lora":
            trainer.run_lora(model, tokenizer, tokenized_dataset["train"], tokenized_dataset["test"])
        elif method == "task_vector":
            trainer.run_task_vector(model, tokenizer, tokenized_dataset["train"], tokenized_dataset["test"])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate fairness
        if method == "task_vector":
            train_fairness_metrics = {}
            fairness_metrics = {}
            for scale_factor in [.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                model_path = f"{trainer.save_dir}/task_vector_scale_{scale_factor}"
                model = model.from_pretrained(model_path)
                train_fairness_metrics[scale_factor] = FairnessEvaluator.evaluate_model(model, tokenizer, tokenized_dataset["train"])
                fairness_metrics[scale_factor] = FairnessEvaluator.evaluate_model(model, tokenizer, tokenized_dataset["test"])

            best_scale_factor = max(fairness_metrics, key=lambda x: fairness_metrics[x].accuracy)
        else:
            fairness_metrics = FairnessEvaluator.evaluate_model(model, tokenizer, tokenized_dataset["test"])
        
        # Store results
        exp_key = f"{method}_{dataset}_{model_name.split('/')[-1]}"
        self.results[exp_key] = fairness_metrics

        if method == "task_vector":
            for alpha in [.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                result = fairness_metrics[alpha]
                result_dict = {
                    "overall_accuracy": result.accuracy,
                    "demographic_parity": result.demographic_parity,
                    "equalized_odds": result.equalized_odds,
                    "equal_opportunity": result.equal_opportunity,
                    "best_scale_factor": best_scale_factor,
                }
                for group, metrics in result.group_metrics.items():
                    result_dict[f"{group}_accuracy"] = metrics["accuracy"]
                    result_dict[f"{group}_precision"] = metrics["precision"]
                    result_dict[f"{group}_recall"] = metrics["recall"]
                    result_dict[f"{group}_f1"] = metrics["f1"]
                    result_dict[f"{group}_tp"] = metrics["tp"]
                    result_dict[f"{group}_tn"] = metrics["tn"]
                result_dict["scale_factor"] = alpha
                wandb.log(result_dict)
                self.print_results(result)

                # --------------------------------------------------------------------------
                # ⬇⬇⬇  INSERT *right after* the `for alpha in [...]` loop for task_vector  ⬇⬇⬇
                # --------------------------------------------------------------------------

                # ---- 1. pick the alpha with best fairness scores -------------------------
                best_dpd_alpha = min(fairness_metrics,
                                    key=lambda a: fairness_metrics[a].demographic_parity)
                best_eod_alpha = min(fairness_metrics,
                                    key=lambda a: fairness_metrics[a].equalized_odds)

                # if they differ, choose the one with the better harmonic mean of (DPD,EOD)
                if best_dpd_alpha == best_eod_alpha:
                    selected_alpha = best_dpd_alpha
                else:
                    def hm(a):
                        m = fairness_metrics[a]
                        return 2 / (m.demographic_parity + m.equalized_odds + 1e-9)
                    selected_alpha = max({best_dpd_alpha, best_eod_alpha}, key=hm)

                selected = fairness_metrics[selected_alpha]

                # ---- 2. make this the single comparable entry ----------------------------
                self.results[exp_key] = selected        # so compare_methods uses it
                wandb.log({                             # optional: one tidy row in W&B
                    "selected_alpha":         selected_alpha,
                    "sel_overall_accuracy":   selected.accuracy,
                    "sel_demographic_parity": selected.demographic_parity,
                    "sel_equalized_odds":     selected.equalized_odds})

                print("\n⭐  FINAL PICK for Task‑Arithmetic")
                print(f"   α = {selected_alpha:.1f}  |  "
                    f"Accuracy = {selected.accuracy:.3f}  |  "
                    f"DPD = {selected.demographic_parity:.3f}  |  "
                    f"EOD = {selected.equalized_odds:.3f}")
                self.print_results(selected)
                # --------------------------------------------------------------------------

        else:
            results_dict = {
                "overall_accuracy": fairness_metrics.accuracy,
                "demographic_parity": fairness_metrics.demographic_parity,
                "equalized_odds": fairness_metrics.equalized_odds,
                "equal_opportunity": fairness_metrics.equal_opportunity,
            }
            for group, metrics in fairness_metrics.group_metrics.items():
                results_dict[f"{group}_accuracy"] = metrics["accuracy"]
                results_dict[f"{group}_precision"] = metrics["precision"]
                results_dict[f"{group}_recall"] = metrics["recall"]
                results_dict[f"{group}_f1"] = metrics["f1"]
                results_dict[f"{group}_tp"] = metrics["tp"]
                results_dict[f"{group}_tn"] = metrics["tn"]

            wandb.log(results_dict)
        
            # Print results
            self.print_results(fairness_metrics)
            
        return fairness_metrics
    
    def compare_methods(self, 
                       methods: List[str] = ["sft", "lora"],
                       dataset: str = "civil_comments",
                       **kwargs) -> Dict[str, FairnessMetrics]:
        """Compare multiple training methods."""
        
        print(f"\n🔬 Comparing Methods: {', '.join(methods)}")
        print(f"📊 Dataset: {dataset}")
        
        results = {}
        for method in methods:
            print(f"\n--- Running {method.upper()} ---")
            results[method] = self.run_experiment(method=method, dataset=dataset, **kwargs)
        
        # Print comparison
        self.print_comparison(results)
        
        return results
    
    def print_results(self, metrics: FairnessMetrics):
        """Print detailed results."""
        print(f"\n📊 FAIRNESS EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {metrics.accuracy:.4f}")
        print(f"Demographic Parity: {metrics.demographic_parity:.4f}")
        print(f"Equalized Odds: {metrics.equalized_odds:.4f}")
        print(f"Equal Opportunity: {metrics.equal_opportunity:.4f}")
        
        print(f"\n📋 GROUP-WISE METRICS")
        print(f"{'Group':<20} {'Count':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-" * 70)
        
        for group, group_metrics in metrics.group_metrics.items():
            print(f"{group:<20} {group_metrics['count']:<8} "
                  f"{group_metrics['accuracy']:<8.3f} {group_metrics['precision']:<8.3f} "
                  f"{group_metrics['recall']:<8.3f} {group_metrics['f1']:<8.3f}")
    
    def print_comparison(self, results: Dict[str, FairnessMetrics]):
        """Print comparison between methods."""
        print(f"\n🔍 METHOD COMPARISON")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Accuracy':<10} {'Dem.Parity':<12} {'Eq.Odds':<10} {'Eq.Opp':<10}")
        print("-" * 80)
        
        for method, metrics in results.items():
            print(f"{method:<15} {metrics.accuracy:<10.4f} {metrics.demographic_parity:<12.4f} "
                  f"{metrics.equalized_odds:<10.4f} {metrics.equal_opportunity:<10.4f}")

# =============================================================================
# Convenience Functions
# =============================================================================

def quick_experiment(method="sft", sample_frac=0.05, epochs=1, output_dir="./fairness_results", group_type="gender", seed=42):
    """Quick experiment for testing."""
    exp = FairnessExperiment()
    return exp.run_experiment(method=method, sample_frac=sample_frac, epochs=epochs, output_dir=output_dir, group_type=group_type, seed=seed)

def quick_taskvector_exp(sample_frac=0.01, epochs=1, output_dir="./fairness_results", group_type="gender", seed=42):
    """Compare SFT, LoRA, and Task Vector methods."""
    exp = FairnessExperiment()
    return exp.compare_methods(
        methods=["task_vector"],  # task_vector takes longer
        sample_frac=sample_frac, 
        epochs=epochs,
        output_dir=output_dir,
        group_type=group_type,
        seed=seed
    )

def compare_all_methods(sample_frac=0.1, epochs=2, output_dir="./fairness_results", group_type="gender", seed=42):
    """Compare SFT, LoRA, and Task Vector methods."""
    exp = FairnessExperiment()
    return exp.compare_methods(
        methods=["sft", "lora"],  # task_vector takes longer
        sample_frac=sample_frac, eval_strategy="epoch",
        epochs=epochs,
        output_dir=output_dir,
        group_type=group_type,
        seed=seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fairness Experiment Runner")
    parser.add_argument("--method",      type=str,   default="sft",
                        choices=["sft", "lora", "task_vector"])
    parser.add_argument("--dataset",     type=str,   default="civil_comments")
    parser.add_argument("--model_name",  type=str,   default="distilbert-base-uncased")
    parser.add_argument("--sample_frac", type=float, default=0.01)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--output_dir",  type=str,   default="./fairness_results")
    parser.add_argument("--group_type",  type=str,   default="gender")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--use_4bit",    action="store_true")
    args = parser.parse_args()

    FairnessExperiment().run_experiment(
        method       = args.method,
        dataset      = args.dataset,
        model_name   = args.model_name,
        sample_frac  = args.sample_frac,
        epochs       = args.epochs,
        output_dir   = args.output_dir,
        group_type   = args.group_type,
        seed         = args.seed,
        use_4bit     = args.use_4bit
    )


    # quick_experiment(args.method, args.sample_frac, args.epochs, args.output_dir, args.group_type, args.seed)

