from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from transformers.models.vit.modeling_vit import ViTModel

from ...utils import BaseModule


class DINOSingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dino-vitb16"
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        # Determine the correct repo ID for the tokenizer/encoder model
        # Use the one specified in its own config section, default to class default if missing.
        tokenizer_repo_id = getattr(self.cfg, 'pretrained_model_name_or_path', "facebook/dino-vitb16") 
        logger.info(f"Image Tokenizer attempting to load config from: {tokenizer_repo_id}") # Added logging
        
        config_path = hf_hub_download(
                repo_id=tokenizer_repo_id, # Use the specific tokenizer repo_id
                filename="config.json",
            )
        
        self.model: ViTModel = ViTModel(
            ViTModel.config_class.from_pretrained(config_path)
        )

        if self.cfg.enable_gradient_checkpointing:
            self.model.encoder.gradient_checkpointing = True

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = self.model(
            rearrange(images, "B N C H W -> (B N) C H W"), interpolate_pos_encoding=True
        )
        local_features, global_features = out.last_hidden_state, out.pooler_output
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
