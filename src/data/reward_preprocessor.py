"""
Preprocessor for Reward Model Training
Keeps prompt, chosen, and rejected fields for preference learning
"""
from src.data.data_loader import DataLoader


class RewardModelProcessor:
    def __init__(self):
        self.dataloader = DataLoader()
        self.raw_train = self.dataloader.train_dataset()
        self.raw_test = self.dataloader.test_dataset()

    def process_one_example(self, example):
        """
        Process a single example - keep prompt, chosen, rejected format
        No formatting needed, just ensure fields exist
        """
        return {
            'prompt': example['prompt'],
            'chosen': example['chosen'],
            'rejected': example['rejected']
        }

    def process_dataset(self, dataset):
        """Process dataset while keeping preference pair structure"""
        # Only keep the required columns
        processed_dataset = dataset.map(
            self.process_one_example,
            remove_columns=[col for col in dataset.column_names
                          if col not in ['prompt', 'chosen', 'rejected']],
            num_proc=4,
            desc="Processing for Reward Model",
        )
        return processed_dataset


if __name__ == "__main__":
    processor = RewardModelProcessor()

    print("Processing training dataset...")
    processed_train = processor.process_dataset(processor.raw_train)
    print(f"✓ Processed {len(processed_train)} training examples")
    print("\nSample example:")
    print(f"Prompt: {processed_train[0]['prompt'][:100]}...")
    print(f"Chosen: {processed_train[0]['chosen'][:100]}...")
    print(f"Rejected: {processed_train[0]['rejected'][:100]}...")

    processed_train.save_to_disk("Dataset/reward_train")
    print("✓ Saved to Dataset/reward_train")

    print("\nProcessing test dataset...")
    processed_test = processor.process_dataset(processor.raw_test)
    print(f"✓ Processed {len(processed_test)} test examples")

    processed_test.save_to_disk("Dataset/reward_test")
    print("✓ Saved to Dataset/reward_test")
