#Data cleaning and tokeization
from typing import Any
from src.data.data_loader import DataLoader


class SFTProcessor:
    def __init__(self):
        self.dataloader = DataLoader()
        self.raw_train = self.dataloader.train_dataset()
        self.raw_test = self.dataloader.test_dataset()

    def format_for_llama(self, prompt: str, chosen: str) -> str:
        """Convert prompt and chosen response to Llama 3.2 chat format."""

        # Remove trailing "Assistant:" if present (it's just a placeholder for the response)
        if prompt.strip().endswith('Assistant:'):
            prompt = prompt.rsplit('Assistant:', 1)[0]

        formatted = "<|begin_of_text|>"

        #replace Human
        text = prompt.replace("Human:", "<|start_header_id|>user<|end_header_id|>")

        #replace Assistant
        text = text.replace("Assistant:", "<|start_header_id|>assistant<|end_header_id|>")

        #add <|eot_id|> after each assistant/user turn
        lines = text.split("<|start_header_id|>")

        formatted_parts = []
        for line in lines:
            if line.strip():
                if "user<|end_header_id|>" in line or "assistant<|end_header_id|>" in line:
                    formatted_parts.append("<|start_header_id|>" + line.strip() + "<|eot_id|>")

        formatted += "".join(formatted_parts)

        #chosen response as the final assistant turn
        formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{chosen.strip()}<|eot_id|>"

        return formatted 

    def process_dataset(self, dataset):
        processed_dataset = dataset.map(
            self.process_one_example,
            remove_columns=dataset.column_names,
            num_proc=4,
            desc="Formatting for Llama 3.2",
        )
        return processed_dataset

    def filter_by_lenght(self):
        pass 

    def process_one_example(self, example):
        """Process a single example from the dataset"""
        # Use the format_for_llama method to convert to Llama format
        formatted_text = self.format_for_llama(example['prompt'], example['chosen'])

        return {'text': formatted_text}
    
    def __call__(self, example) -> Any:
        text = self.process_one_example(example)
            

if __name__ == "__main__":
    from pprint import pprint

    sft = SFTProcessor()
    processed_train = sft.process_dataset(sft.raw_train)
    print(f"Processed {len(processed_train)} examples")
    print("\nSample formatted text:")
    # print(processed_train[0]['text'])    
    processed_train.save_to_disk("Dataset/train")
    processed_test = sft.process_dataset(sft.raw_test)
    print(f"Processed {len(processed_test)} examples")
    print("\nSample formatted text:")
    # print(processed_test[0]['text'])
    processed_test.save_to_disk("Dataset/test")