import torch


class RewardConfigs:
    def __init__(self):
        # Model settings
        self.model_path = "./checkpoints/sft_model"  # Path to SFT checkpoint
        self.torch_dtype = torch.float16
        self.device = "auto"

        # Dataset paths - use reward-specific datasets
        self.train_path = "Dataset/reward_train"
        self.eval_path = "Dataset/reward_test"

        # Training hyperparameters - OPTIMIZED FOR 12GB GPU
        self.batch_size = 1  # Reduced from 2 to save memory
        self.gradient_accumulation_steps = 4  # Effective batch size = 4
        self.epochs = 1
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.warmup_steps = 0
        self.max_length = 256  # Reduced from 512 to save memory
        self.log_steps = 50
        self.eval_steps = 2000
        self.save_steps = 2000

        # Output directory
        self.output_dir = "./checkpoints/reward_model"

        # Mixed precision
        self.use_amp = True

        # Memory optimization flags
        self.gradient_checkpointing = True  # Enable gradient checkpointing
        self.use_8bit = False  # Set to True if you want to use 8-bit quantization  