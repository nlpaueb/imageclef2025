import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from .refiner_base import CaptionRefiner

class LLaMACaptionRefiner(CaptionRefiner):
    def __init__(self, model_config):
        model_id = model_config["model_id"]
        access_token = model_config["token"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            quantization_config=quant_config,
            use_auth_token=access_token,
        )
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="cuda")
        self.eos_token_id = self.tokenizer.eos_token_id

        self.instruction = (
            "Refine the caption into a concise sentence (max 50 words)... [truncated for brevity]"
        )

    def refine_caption(self, image, draft_caption, neighbor_image=None, neighbor_caption=None):
        messages = [
            {"role": "system", "content": "You are an experienced medical professional assistant."},
            {"role": "image", "content": image},
            {"role": "user", "content": f"Draft Caption: {draft_caption}"}
        ]

        if neighbor_caption:
            messages.append({
                "role": "user",
                "content": f"Neighbor Caption: {neighbor_caption}\n\n{self.instruction}"
            })
        else:
            messages.append({"role": "user", "content": self.instruction})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(prompt, max_new_tokens=112, eos_token_id=[self.eos_token_id], do_sample=False, num_beams=1)

        generated_text = outputs[0]["generated_text"]
        return self._clean_text(generated_text, prompt)

    def _clean_text(self, text, prompt):
        clean = text[len(prompt):].strip()
        clean = re.sub(r'<\|.*?\|>', '', clean)
        clean = re.sub(r'^\s*Refined\s*caption:\s*', '', clean, flags=re.IGNORECASE).strip()
        return clean