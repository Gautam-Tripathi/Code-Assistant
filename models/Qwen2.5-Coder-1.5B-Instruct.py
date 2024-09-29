from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

class QwenCoder:

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        self.quant_config = QuantoConfig(weights='int8')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=self.quant_config
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def inference(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response