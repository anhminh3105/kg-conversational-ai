"""
Train NLP to Cypher Model
Direct question-to-Cypher translation using T5-small (lightweight, CPU-friendly)
"""
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings('ignore')

class CypherDataset(Dataset):
    """Dataset for question-to-Cypher pairs"""
    
    def __init__(self, data_path, tokenizer, max_source_length=128, max_target_length=256):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Filter examples with Cypher queries
        self.examples = [ex for ex in self.data if 'cypher' in ex]
        
        # Augment data by creating variations
        self.augmented_examples = self._augment_data()
        
        print(f"✓ Loaded {len(self.examples)} base examples")
        print(f"✓ Generated {len(self.augmented_examples)} augmented examples")
        print(f"✓ Total training examples: {len(self.augmented_examples)}")
    
    def _augment_data(self):
        """Create augmented examples by varying the Cypher queries"""
        augmented = []
        
        for ex in self.examples:
            question = ex['question']
            cypher = ex['cypher']
            
            # Add original
            augmented.append({'question': question, 'cypher': cypher})
            
            # Create variations with different RETURN clauses
            if 'RETURN' in cypher:
                base_query = cypher.split('RETURN')[0]
                
                # Variation 1: Simplified RETURN
                simple_cypher = base_query + "RETURN *"
                augmented.append({'question': question, 'cypher': simple_cypher})
                
                # Variation 2: Add lowercase version of question
                augmented.append({'question': question.lower(), 'cypher': cypher})
                
                # Variation 3: Add question mark if missing
                if not question.endswith('?'):
                    augmented.append({'question': question + '?', 'cypher': cypher})
        
        return augmented
    
    def __len__(self):
        return len(self.augmented_examples)
    
    def __getitem__(self, idx):
        example = self.augmented_examples[idx]
        
        # Prepare input: "translate to cypher: {question}"
        input_text = f"translate to cypher: {example['question']}"
        target_text = example['cypher']
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


def train_model(
    data_path='data/training_data_complete.json',
    model_name='t5-small',
    output_dir='models/question_to_cypher',
    epochs=10,
    batch_size=4,
    learning_rate=3e-4
):
    """
    Train T5 model for question-to-Cypher translation
    
    Args:
        data_path: Path to training data JSON
        model_name: Hugging Face model name (default: t5-small for CPU)
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    
    print("\n" + "="*70)
    print("TRAINING QUESTION-TO-CYPHER MODEL")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print("✓ Model loaded\n")
    
    # Create dataset
    print("Preparing dataset...")
    dataset = CypherDataset(data_path, tokenizer)
    
    # Split into train/validation (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"✓ Train examples: {train_size}")
    print(f"✓ Validation examples: {val_size}\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',  # Disable wandb
        no_cuda=not torch.cuda.is_available(),  # Use CPU if no GPU
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("Starting training...")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("-"*70)
    
    trainer.train()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Model saved successfully!")
    
    # Evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print(f"✓ Validation Loss: {eval_results['eval_loss']:.4f}")
    
    print("\n" + "="*70)
    print("Ready to test! Run: python scripts/test_model.py")
    print("="*70 + "\n")
    
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Question-to-Cypher model')
    parser.add_argument('--data', default='data/training_data_complete.json', help='Training data path')
    parser.add_argument('--model', default='t5-small', help='Base model name')
    parser.add_argument('--output', default='models/question_to_cypher', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
