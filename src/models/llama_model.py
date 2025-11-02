import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LlamaModel:
    def __init__(self, model_id:str):
        self.model = self._load_model(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def _load_model(self, model_id:str):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto"
        )
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



# if __name__ == "__main__":
#     model_id = "meta-llama/Llama-3.2-3B-Instruct"

#     prompt = """<|start_header_id|>user<|end_header_id|>
#     Explain what attention mechanism is.
#     <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """

#     llama_model = LlamaModel(model_id)
#     generated_text = llama_model.generate_text(prompt)
#     print("Generated Text:", generated_text)