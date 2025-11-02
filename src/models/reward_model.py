from transformers import Auto
class RewardModel:
    def __init__(self):
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "./checkpoints/sft_model",  # Your SFT checkpoint
            num_labels=1  # Regression: output single score
        )
        