import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, AutoPeftModelForCausalLM


class LlamaModel:
    def __init__(self, model_id: str, 
                 checkpoint_path: str ,
                 use_checkpoint: bool = False):
        self.use_checkpoint = use_checkpoint
        self.checkpoint_path = checkpoint_path or model_id

        if use_checkpoint:
            self.model = self._load_checkpoint_model(self.checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        else:
            self.model = self._load_model(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _load_model(self, model_id: str):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return model

    def _load_checkpoint_model(self, checkpoint_path: str):
        print(f"Loading fine-tuned model from: {checkpoint_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            is_trainable=False  # for the inference mode
        )
        print("✓ LoRA adapter loaded successfully!")
        return model
    
    def generate_text(self, prompt:str, max_new_tokens:int=200, temperature:float=1):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        output = pipe(prompt)[0]["generated_text"]
        return output[len(prompt):]



if __name__ == "__main__":
    checkpoint_path = "./checkpoints/sft_model"
    llama_model = LlamaModel(
        model_id="meta-llama/Llama-3.2-3B",         
        use_checkpoint=True, 
        checkpoint_path=checkpoint_path
    )


    prompt = """<|start_header_id|>user<|end_header_id|>
    Explain what attention mechanism is.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    generated_text = llama_model.generate_text(
        prompt,
        max_new_tokens=200,
        temperature=0.7  #temperature = more focused, less random
    )
    print("Generated Text:", generated_text)