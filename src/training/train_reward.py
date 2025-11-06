import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, get_scheduler

from src.configs.reward_configs import RewardConfigs
from src.data.data_utils import PreferencePairDataset, collate_fn


class RewardModel(nn.Module):
    """Wrapper that adds a reward head to the base language model."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        hidden_size = base_model.config.hidden_size
        
        #linear head that outputs a single scalar reward
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        #last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        #reward logits for all positions
        rewards = self.reward_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        #reward at the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_size = rewards.shape[0]
        
        #the reward at the last position
        final_rewards = rewards[torch.arange(batch_size, device=rewards.device), sequence_lengths]
        
        return final_rewards  # [batch_size]


class RewardModelTrainer:
    def __init__(self, configs:RewardConfigs):
        self.configs = configs
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_dataset(self):
        """Load train and eval datasets from disk."""
        self.train_dataset = load_from_disk(dataset_path=self.configs.train_path)
        self.eval_dataset = load_from_disk(dataset_path=self.configs.eval_path)
        print(f"Loaded {len(self.train_dataset)} training samples")
        print(f"Loaded {len(self.eval_dataset)} eval samples")

    def init_model_and_tokenizer(self, checkpoint_path: str = "./checkpoints/sft_model"):
        """Initialize the reward model from a fine-tuned SFT checkpoint."""
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        #wraping with reward head
        reward_model = RewardModel(base_model)
        reward_model = reward_model.to(self.device)
        
        #gradient checkpointing to save memory
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        
        return reward_model

    @staticmethod
    def pairwise_loss(chosen_rewards, rejected_rewards, epsilon: float = 1e-12):
        """
        Compute pairwise ranking loss.
        We want chosen_rewards > rejected_rewards.
        Loss = -log(sigmoid(chosen - rejected))
        """
        difference = chosen_rewards - rejected_rewards
        loss = -torch.log(torch.sigmoid(difference) + epsilon).mean()
        return loss

    def evaluate(self, eval_loader):
        """Evaluate pairwise accuracy on the eval set."""
        self.model.eval()
        correct = 0
        total = 0

        # Use AMP during evaluation to match training dtype
        use_amp = getattr(self.configs, "use_amp", True) and torch.cuda.is_available()

        with torch.no_grad():
            for batch in eval_loader:
                # to cuda
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)

                #rewards with autocast to handle dtype consistency
                if use_amp:
                    with torch.cuda.amp.autocast():
                        chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                        rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                else:
                    chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                    rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                
                # Count how many times chosen > rejected
                correct += (chosen_rewards > rejected_rewards).sum().item()
                total += len(chosen_rewards)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def train(self):
        """Main training loop for the reward model."""
        # 1) Load datasets
        self.load_dataset()

        # 2) Initialize tokenizer & model
        self.model = self.init_model_and_tokenizer(
            checkpoint_path=getattr(self.configs, "sft_checkpoint", "./checkpoints/sft_model")
        )

        # 3) Wrap datasets with PreferencePairDataset
        max_length = getattr(self.configs, "max_length", 512)
        batch_size = getattr(self.configs, "batch_size", 2)
        
        train_ds = PreferencePairDataset(
            self.train_dataset, 
            self.tokenizer, 
            max_length=max_length
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        eval_loader = None
        if self.eval_dataset is not None:
            eval_ds = PreferencePairDataset(
                self.eval_dataset, 
                self.tokenizer, 
                max_length=max_length
            )
            eval_loader = DataLoader(
                eval_ds, 
                batch_size=batch_size, 
                collate_fn=collate_fn
            )

        # 4) Optimizer and scheduler
        lr = getattr(self.configs, "reward_lr", 1e-5)
        weight_decay = getattr(self.configs, "weight_decay", 0.01)
        
        optimizer = AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        epochs = getattr(self.configs, "epochs", 1)
        total_steps = len(train_loader) * epochs
        warmup_steps = getattr(self.configs, "warmup_steps", 0)
        
        scheduler = get_scheduler(
            name=getattr(self.configs, "lr_scheduler_type", "linear"),
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        #mixed precision training
        use_amp = getattr(self.configs, "use_amp", True) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # 5) Training loop
        print("\n" + "="*80)
        print("Starting Reward Model Training")
        print("="*80 + "\n")
        
        global_step = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            correct_predictions = 0
            total_pairs = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                #batch to cuda
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                optimizer.zero_grad()
                
                #forward pass with mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                        rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                        loss = self.pairwise_loss(chosen_rewards, rejected_rewards)
                else:
                    chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                    rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                    loss = self.pairwise_loss(chosen_rewards, rejected_rewards)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                #batch accuracy
                with torch.no_grad():
                    correct_predictions += (chosen_rewards > rejected_rewards).sum().item()
                    total_pairs += len(chosen_rewards)
                
                #updating progress bar
                if batch_idx % 10 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    batch_acc = correct_predictions / total_pairs if total_pairs > 0 else 0.0
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'acc': f'{batch_acc:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_acc = correct_predictions / total_pairs if total_pairs > 0 else 0.0
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{epochs} Complete")
            print(f"{'='*80}")
            print(f"Average Loss: {avg_epoch_loss:.4f}")
            print(f"Train Pairwise Accuracy: {train_acc:.4f}")
            
            if eval_loader is not None:
                print("\nRunning evaluation...")
                eval_acc = self.evaluate(eval_loader)
                print(f"Eval Pairwise Accuracy: {eval_acc:.4f}")
                print(f"{'='*80}\n")

        # 6) Save model and tokenizer
        out_dir = getattr(
            self.configs,
            "reward_output_dir",
            getattr(self.configs, "output_dir", "./checkpoints/reward_model")
        )
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nSaving reward model to {out_dir}...")

        #save PEFT model (LoRA adapters)
        self.model.base_model.save_pretrained(out_dir)

        #save only the reward head weights separately
        reward_head_state = {
            'reward_head.weight': self.model.reward_head.weight.data.cpu()
        }
        torch.save(reward_head_state, os.path.join(out_dir, "reward_head.pt"))

        #save tokenizer
        self.tokenizer.save_pretrained(out_dir)

        config_dict = vars(self.configs).copy()

        # Convert non-serializable objects to strings
        for key, value in config_dict.items():
            if hasattr(value, '__name__'):  # Functions, classes, types
                config_dict[key] = str(value)
            elif isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_dict[key] = str(value)

        with open(os.path.join(out_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        print("\n" + "="*80)
        print("Reward Model Saved Successfully!")
        print("="*80)
        print(f"Location: {out_dir}")
        print(f"  - Base model + LoRA: adapter_config.json, adapter_model.bin")
        print(f"  - Reward head: reward_head.pt")
        print(f"  - Tokenizer: tokenizer_config.json, tokenizer.json")
        print(f"  - Training config: training_config.json")
        
        return self.model
    

if __name__ == "__main__":
    reward_configs = RewardConfigs()

    r_train = RewardModelTrainer(configs=reward_configs)
    r_train.train()