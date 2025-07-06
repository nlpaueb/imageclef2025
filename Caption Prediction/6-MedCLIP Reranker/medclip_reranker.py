import torch
from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModelViT
from PIL import Image

class MedCLIPReranker:
    def __init__(self, checkpoint_path, device='cuda:0'):
        self.device = device
        print("Initializing MedCLIP reranker...")
        self.processor = MedCLIPProcessor()
        self.model = MedCLIPModel(
            vision_cls=MedCLIPVisionModelViT,
            checkpoint=checkpoint_path
        )
        self.model.from_pretrained()
        self.model.to(self.device)
        self.model.eval()

    def rank_captions(self, image, captions):
        """
        Ranks captions by similarity to image and returns the best one.
        :param image: PIL.Image
        :param captions: list of str
        :return: best caption (str)
        """
        inputs = self.processor(
            text=captions,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs['pixel_values'] = inputs['pixel_values'].permute(0, 3, 1, 2)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits'].squeeze(0)
            probs = torch.softmax(logits, dim=0)

        best_idx = probs.argmax().item()
        return captions[best_idx], probs.cpu().numpy()
