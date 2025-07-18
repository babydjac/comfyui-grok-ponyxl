import requests
import base64
from PIL import Image
import io
import numpy as np

class GrokPonyXLPrompter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grok_api_key": ("STRING", {
                    "multiline": False,
                    "placeholder": "Paste your Grok API key here"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "prompt_from_image"
    CATEGORY = "prompt"

    def prompt_from_image(self, image, grok_api_key):
        pil_image = self.tensor_to_pil(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        system_prompt = (
            "You are an expert Stable Diffusion prompter specializing in PonyXL and NSFW realism. "
            "Your task: Given an image, write an optimal ComfyUI prompt string using concise Danbooru tags, weighted photo terms, and "
            "details that maximize human-realistic quality in PonyXL NSFW generations. Format for direct input into ComfyUI's text prompt field. "
            "Start the prompt with: 'score_9, score_8_up, score_7_up, score_6_up,' then Danbooru-style tags, weighted realism terms, and "
            "no negative prompt. DO NOT add extra comments. Return only the prompt."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": img_b64, "image_type": "jpg"},
                    {"type": "text", "text": "Write a PonyXL NSFW human-realistic prompt for this image, per instructions."}
                ]}
            ],
            "model": "grok-2-vision-latest",
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {grok_api_key}",
            "Content-Type": "application/json"
        }
        url = "https://api.x.ai/v1/chat/completions"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            prompt = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            prompt = f"ERROR: {str(e)}"
        return (prompt,)

    def tensor_to_pil(self, tensor):
        import numpy as np

        arr = tensor
        # If it's a list or tuple, take first until it's np.ndarray
        while isinstance(arr, (list, tuple)):
            arr = arr[0]
        # To numpy
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr)
        # Squeeze all singleton dimensions
        arr = np.squeeze(arr)
        # Now handle shape
        if arr.ndim == 1:
            raise ValueError("Input image has only one dimension, not valid as image.")
        # Make (H, W, 3)
        if arr.ndim == 3:
            if arr.shape[0] == 3 and arr.shape[2] != 3:
                arr = arr.transpose(1,2,0)
            elif arr.shape[-1] == 3:
                pass
            else:
                if arr.shape[0] == 1:
                    arr = arr[0, :, :]
                elif arr.shape[-1] == 1:
                    arr = arr[:, :, 0]
        # Expand grayscale to RGB
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        # Dtype fix
        if arr.dtype in [np.float32, np.float64]:
            arr = (arr * 255).clip(0,255).astype('uint8')
        elif arr.dtype != np.uint8:
            arr = arr.astype('uint8')
        if arr.shape[-1] != 3:
            raise ValueError(f"Cannot handle image tensor shape: {arr.shape}, dtype: {arr.dtype}")
        return Image.fromarray(arr)
