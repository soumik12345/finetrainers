from typing import Dict, List, Optional, Union

import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from torch.nn.modules import Module
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from finetrainers.models.wan.base_specification import WanModelSpecification
from finetrainers.processors.base import ProcessorMixin
from finetrainers.utils import get_non_null_items


class Wan22ModelSpecification(WanModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        transformer_2_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        transformer_2_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            transformer_2_id=transformer_2_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            transformer_2_dtype=transformer_2_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

    def load_diffusion_models(self) -> Dict[str, Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        diffusion_model_components = super().load_diffusion_models()

        if self.transformer_2_id is not None:
            transformer_2 = WanTransformer3DModel.from_pretrained(
                self.transformer_2_id, torch_dtype=self.transformer_2_dtype, **common_kwargs
            )
        else:
            transformer_2 = WanTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer_2",
                torch_dtype=self.transformer_2_dtype,
                **common_kwargs,
            )

        diffusion_model_components["transformer_2"] = transformer_2
        return diffusion_model_components

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        vae: Optional[AutoencoderKLWan] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        image_encoder: Optional[CLIPVisionModel] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Union[WanPipeline, WanImageToVideoPipeline]:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "transformer_2": transformer_2,
            "vae": vae,
            "scheduler": scheduler,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }
        components = get_non_null_items(components)

        if self.transformer_config.get("image_dim", None) is not None:
            pipe = WanPipeline.from_pretrained(
                self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(
                self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
            )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        if not training:
            pipe.transformer.to(self.transformer_dtype)

        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
