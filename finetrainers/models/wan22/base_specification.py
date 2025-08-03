from typing import Dict, List, Optional

import torch
from diffusers import AutoencoderKLWan
from torch.nn.modules import Module
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.models.wan.base_specification import (
    WanImageConditioningLatentEncodeProcessor,
    WanImageEncodeProcessor,
    WanLatentEncodeProcessor,
)
from finetrainers.processors import T5Processor
from finetrainers.processors.base import ProcessorMixin


class Wan22ModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        use_last_frame = self.transformer_config.get("pos_embed_seq_len", None) is not None

        self.condition_model_processors = (
            [T5Processor(["encoder_hidden_states", "__drop__"])]
            if condition_model_processors is None
            else condition_model_processors
        )

        latent_model_processors = (
            [WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"])]
            if latent_model_processors is None
            else latent_model_processors
        )
        if self.transformer_config.get("image_dim", None) is not None:
            latent_model_processors.append(
                WanImageConditioningLatentEncodeProcessor(
                    ["latent_condition", "__drop__", "__drop__", "latent_condition_mask"],
                    use_last_frame=use_last_frame,
                )
            )
            latent_model_processors.append(
                WanImageEncodeProcessor(["encoder_hidden_states_image"], use_last_frame=use_last_frame)
            )

        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        tokenizer_kwargs = {
            "pretrained_model_name_or_path": self.tokenizer_id
            if self.tokenizer_id is not None
            else self.pretrained_model_name_or_path,
            **common_kwargs,
        }
        if self.tokenizer_id is None:
            tokenizer_kwargs["subfolder"] = "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        vae_kwargs = {
            "pretrained_model_name_or_path": self.vae_id
            if self.vae_id is not None
            else self.pretrained_model_name_or_path,
            "torch_dtype": self.vae_dtype,
            **common_kwargs,
        }
        if self.vae_id is None:
            vae_kwargs["subfolder"] = "vae"
        vae = AutoencoderKLWan.from_pretrained(**vae_kwargs)

        models = {"vae": vae}

        if self.transformer_config.get("image_dim", None) is not None:
            image_encoder = CLIPVisionModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=torch.bfloat16
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_processor"
            )
            models["image_encoder"] = image_encoder
            models["image_processor"] = image_processor

        return models
