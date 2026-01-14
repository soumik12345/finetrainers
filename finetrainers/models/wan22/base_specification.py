from typing import Dict, List, Optional, Union, Tuple

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

import finetrainers.functional as FF
from finetrainers.models.wan.base_specification import WanModelSpecification
from finetrainers.processors.base import ProcessorMixin
from finetrainers.utils import get_non_null_items

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


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
        boundary_ratio: float = 0.875,
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
            boundary_ratio=boundary_ratio
        )
        self.boundary_ratio = boundary_ratio

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

        scheduler = FlowMatchEulerDiscreteScheduler()

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
            pipe.transformer_2.to(self.transformer_2_dtype)

        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

    def forward(
        self,
        transformer_1: WanTransformer3DModel,
        transformer_2: WanTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False  # See explanation in prepare_latents
        latent_condition = latent_condition_mask = None

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)
        else:
            latents = latent_model_conditions.pop("latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=generator)

            if latent_condition is not None:
                mu, logvar = torch.chunk(latent_condition, 2, dim=1)
                mu = self._normalize_latents(mu, latents_mean, latents_std)
                logvar = self._normalize_latents(logvar, latents_mean, latents_std)
                latent_condition = torch.cat([mu, logvar], dim=1)

                posterior = DiagonalGaussianDistribution(latent_condition)
                latent_condition = posterior.mode()

            del posterior

        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        timesteps = (sigmas.flatten() * 1000.0).long()

        if self.transformer_config.get("image_dim", None) is not None:
            noisy_latents = torch.cat([noisy_latents, latent_condition_mask, latent_condition], dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        boundary_timestep = self.boundary_ratio * 1000 # TODO: Replace with self.scheduler.config.num_train_timesteps

        for t in timesteps:
            self._current_timestep = t
            if boundary_timestep is None or t >= boundary_timestep:
                # wan2.1 or high-noise stage in wan2.2
                current_model = transformer_1
            else:
                # low-noise stage in wan2.2
                current_model = transformer_2

            pred = current_model(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0]

            target = FF.flow_match_target(noise, latents)
            self._current_timestep = None

        return pred, target, sigmas