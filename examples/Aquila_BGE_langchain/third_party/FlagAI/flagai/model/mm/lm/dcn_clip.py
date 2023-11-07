import torch
import os
import torch.nn as nn
from transformers import AltCLIPProcessor
from flagai.model.mm.modeling_altclip import AltCLIPModel

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    

class DCNCLIP30M1024(AbstractEncoder):
    def __init__(self, device="cuda", max_length=77, model_name=None, download_path=None):
        super().__init__()
        self.device = device
        self.max_length = max_length
        ckpt_path = os.path.join(download_path, model_name)
        self.ch_clip_model = AltCLIPModel.from_pretrained(ckpt_path=ckpt_path)
        # print("^_^ Using the right lm model!!!")
        self.ch_clip_model = self.ch_clip_model.eval()
        print("Language model Loaded!!!^_^")

        for param in self.ch_clip_model.parameters():
            param.requires_grad = False
        ckpt_path = os.path.join(download_path, model_name)
        self.processor = AltCLIPProcessor.from_pretrained(ckpt_path)
        self.tokenizer = self.processor.tokenizer

        self.text_encoder = self.ch_clip_model.text_model
        
    def forward(self, text):
        
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=False,return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.text_encoder.device)
        
        z = self.text_encoder(**tokens)['penultimate_hidden_state']

        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        return z
