import torch
from torch.utils.data import DataLoader, Dataset


class PreferencePairDataset(Dataset):
    """
    A small wrapper that expects dataset with fields:
    - prompt (str)
    - chosen (str)
    - rejected (str)
    """
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Combine prompt + response for full context
        chosen_text = prompt + " " + chosen
        rejected_text = prompt + " " + rejected

        chosen_inputs = self.tokenizer(
            chosen_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected_inputs = self.tokenizer(
            rejected_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_inputs["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_inputs["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_inputs["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_inputs["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    """Collate function to batch multiple examples"""
    return {
        "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]),
        "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]),
        "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]),
        "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]),
    }
