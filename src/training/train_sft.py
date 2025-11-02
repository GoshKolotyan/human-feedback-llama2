import torch

from datasets import load_from_disk

from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments


from src.data.data_loader import DataLoader
from src.models.llama_model import LlamaModel
from src.configs.sft_configs import SFTConfigs



class SupervisedFineTuningTrainer:
    def __init__(self, config):
        self.configs = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None 
        self.eval_dataset = None
        pass 

    def load_dataset(self):
        self.train_dataset = load_from_disk(self.configs.train_path)
        self.eval_dataset = load_from_disk(self.configs.eval_path)

    def load_model(self):
        lora_config = LoraConfig(
                r=self.configs.lora_r,                    # Rank (16)
                lora_alpha=self.configs.lora_alpha,       # Alpha (32)
                target_modules=self.configs.target_modules, # Which layers
                lora_dropout=self.configs.lora_dropout,   # Dropout (0.05)
                bias="none",                              # Don't train bias
                task_type="CAUSAL_LM"                     # Task: language modeling
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs.model_id,
            dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs.model_id)

        # Configure tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = get_peft_model(self.model, lora_config)

        # Enable gradient checkpointing to save memory
        if self.configs.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for memory savings")

        self.model.print_trainable_parameters()


        return self.model

    def setup_training_args(self):
        """Configure training arguments from config."""
        
        training_args = TrainingArguments(
            # Output and saving
            output_dir=self.configs.output_dir,
            logging_dir=self.configs.logging_dir,
            save_strategy=self.configs.save_strategy,
            save_total_limit=self.configs.save_total_limit,
            
            # Training hyperparameters
            num_train_epochs=self.configs.num_train_epochs,
            per_device_train_batch_size=self.configs.per_device_train_batch_size,
            gradient_accumulation_steps=self.configs.gradient_accumulation_steps,
            
            # Learning rate and optimization
            learning_rate=self.configs.learning_rate,
            warmup_steps=self.configs.warmup_steps,
            weight_decay=self.configs.weight_decay,
            max_grad_norm=self.configs.max_grad_norm,
            optim=self.configs.optim,
            lr_scheduler_type=self.configs.lr_scheduler_type,
            
            # Memory optimization
            fp16=self.configs.fp16,
            gradient_checkpointing=self.configs.gradient_checkpointing,
            
            # Logging and evaluation
            logging_steps=self.configs.logging_steps,
            eval_strategy=self.configs.evaluation_strategy,
            eval_steps=self.configs.eval_steps,
            
            # Other settings
            report_to="none",  # Don't report to wandb/tensorboard (can change later)
            push_to_hub=False,  # Don't push to HuggingFace Hub
        )
        
        print("Training arguments configured!")
        return training_args

    def train(self):
        """Main training method - orchestrates everything."""        
        print("="*80)
        print("Starting SFT Training")
        print("="*80)
        
        # Step 1: Load model with LoRA
        print("\n[1/6] Loading model with LoRA...")
        self.load_model()
        
        # Step 2: Load datasets
        print("\n[2/6] Loading datasets...")
        self.load_dataset()
        print(f"  - Train examples: {len(self.train_dataset)}")
        print(f"  - Eval examples: {len(self.eval_dataset)}")
        
        # Step 3: Setup training arguments
        print("\n[3/6] Setting up training arguments...")
        training_args = self.setup_training_args()
        
        # Step 4: Create SFTTrainer
        print("\n[4/6] Creating SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            processing_class=self.tokenizer,
            # max_seq_length=self.configs.max_seq_length,  # Fixed: uncommented
            # packing=False,  # Fixed: uncommented - don't pack examples
        )
        
        # Step 5: Start training!
        print("\n[5/6] Starting training...")
        print("="*80)
        trainer.train()
        
        # Step 6: Save final model
        print("\n[6/6] Saving final model...")
        trainer.save_model(self.configs.output_dir)
        self.tokenizer.save_pretrained(self.configs.output_dir)
        
        print("="*80)
        print(f"✅ Training complete! Model saved to: {self.configs.output_dir}")
        print("="*80)
        
        return trainer


if __name__ == "__main__":
    # Create an instance of YOUR config class
    configs = SFTConfigs()
    
    # Create trainer with your config
    sft = SupervisedFineTuningTrainer(configs)
    
    # Load model with LoRA
    # sft.load_model()

    #Load Dataset 
    # sft.load_dataset()
    
    #setup arg
    # sft.setup_training_args()

    sft.train()
    # print("Model loaded successfully!")