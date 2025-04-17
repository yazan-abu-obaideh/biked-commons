import abc

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizerFast, CLIPModel
from torchvision import transforms


#TODO investigate memory footprint of loading these here. Perhaps avoid?
_DEVICE = "cuda"
_MODEL_ID = "openai/clip-vit-base-patch32"
_CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_ID)
_TOKENIZER = CLIPTokenizerFast.from_pretrained(_MODEL_ID)
_MODEL = CLIPModel.from_pretrained(_MODEL_ID)  # .to(device)


class ClipEmbeddingCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def from_text(self, text: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def from_image_path(self, image_path: str) -> np.ndarray:
        pass


class ClipEmbeddingCalculatorImpl(ClipEmbeddingCalculator):
    def from_text(self, text: str) -> np.ndarray:
        embedding_tensor = _MODEL.get_text_features(**_TOKENIZER(text, return_tensors="pt"))
        return embedding_tensor

    def from_image_path(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        img = img.resize((width // 2, height // 2))
        result = Image.new(img.mode, (1300, 1300), (255, 255, 255))
        result.paste(img, img.getbbox())
        image = np.asarray(result)
        img_processed = _CLIP_PROCESSOR(text=None, images=image, return_tensors='pt')['pixel_values']  # .to(device)
        embedding_tensor = _MODEL.get_image_features(img_processed)
        return embedding_tensor
    
    def from_image_tensor(self, image_tensor: torch.Tensor) -> np.ndarray:
        img = image_tensor.cpu().clone()
        img = transforms.Resize((1300, 1300))(img)
        result = Image.new(img.mode, (1300, 1300), (255, 255, 255))
        result.paste(img, img.getbbox())
        image = np.asarray(result)
        img_processed = _CLIP_PROCESSOR(text=None, images=image, return_tensors='pt')['pixel_values']
        embedding_tensor = _MODEL.get_image_features(img_processed)
        return embedding_tensor

def get_augmented_views_gpu(images_tensor):
    transform = transforms.RandomApply([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomAdjustSharpness(0.2), 
                                        transforms.RandomAdjustSharpness(2), 
                                        transforms.RandomPerspective(fill=(0, 0, 0)),
                                        transforms.RandomRotation(degrees = 45, fill= (0, 0, 0)), 
                                       #  transforms.ColorJitter(brightness=0.1, contrast = 0.1, saturation=0.1, hue=0.0),
                                       ],p=1)
    res = transform(images_tensor.cuda()).cpu()
    return res


