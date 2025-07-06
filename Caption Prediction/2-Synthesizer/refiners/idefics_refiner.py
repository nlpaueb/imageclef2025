import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from .refiner_base import CaptionRefiner

class IdeficsCaptionRefiner(CaptionRefiner):
    def __init__(self, model_config):
        self.processor = AutoProcessor.from_pretrained(model_config["model_id"])
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_config["model_id"],
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            device_map="auto"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_base = (
            "You are assisting with the refinement of medical image captions... [truncated for brevity]"
        )

    def refine_caption(self, image, draft_caption, neighbor_image=None, neighbor_caption=None):
        images = [image]
        if neighbor_image:
            images.append(neighbor_image)

        text_content = f"Draft Caption: {draft_caption}\n"
        if neighbor_caption:
            text_content += f"Neighbor Caption: {neighbor_caption}\n"
        text_content += self.prompt_base

        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text_content}]
        }]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False
            )

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return decoded.split("Assistant:")[-1].strip()