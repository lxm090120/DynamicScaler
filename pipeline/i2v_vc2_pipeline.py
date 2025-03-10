import os
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from tqdm import trange
from typing import List, Optional, Union, Dict, Any
from diffusers import logging

from diffusers import DiffusionPipeline
from lvdm.models.ddpm3d import LatentVisualDiffusion
from pipeline.vc2_lvdm_scheduler import lvdm_DDIM_Scheduler
from utils.precast_latent_utils import encode_images_list_to_latent_tensor
from utils.shift_window_utils import RingLatent, RingImageTensor
from utils.tensor_utils import mix_latents_with_mask

# from diffusers.schedulers.scheduling_ddim import DDIMScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VC2_Pipeline_I2V(DiffusionPipeline):       # TODO : 添加 Dynamic Crafter 支持
    def __init__(
            self,
            pretrained_t2v: LatentVisualDiffusion,
            scheduler: lvdm_DDIM_Scheduler,
            model_config: Dict[str, Any] = None,
    ):
        super().__init__()

        self.scheduler: lvdm_DDIM_Scheduler
        self.pretrained_t2v: LatentVisualDiffusion

        self.register_modules(
            pretrained_t2v=pretrained_t2v,
            scheduler=scheduler,
        )
        self.vae = pretrained_t2v.first_stage_model
        self.unet = pretrained_t2v.model.diffusion_model
        self.text_encoder = pretrained_t2v.cond_stage_model

        self.model_config = model_config
        self.vae_scale_factor = 8


    def _load_imgs_from_paths(self, img_path_list: list, height=320, width=512):
        batch_tensor = []
        for filepath in img_path_list:
            _, filename = os.path.split(filepath)
            _, ext = os.path.splitext(filename)
            if ext == '.png' or ext == '.jpg':
                img = Image.open(filepath).convert("RGB")
                rgb_img = np.array(img, np.float32)
                rgb_img = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1) # .float()
            else:
                print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
                raise NotImplementedError
            img_tensor = (img_tensor / 255. - 0.5) * 2
            batch_tensor.append(img_tensor)
        return torch.stack(batch_tensor, dim=0)





    @torch.no_grad()
    def basic_sample(
            self,
            prompt: Union[str, List[str]] = None,
            img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            # lcm_origin_steps: int = 50,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            skip_time_step_idx=None,
            **kwargs
    ):
        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if isinstance(img_cond_path, str):
                img_cond_path = [img_cond_path]
            else:
                assert len(img_cond_path) == 1, "[basic_sample] cond img should have same amount as text prompts"
        elif prompt is not None and isinstance(prompt, list):
            assert len(prompt) == len(img_cond_path), "[basic_sample] cond img should have same amount as text prompts"
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # do_classifier_free_guidance = guidance_scale > 0.0  # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

        # 3. Encode input prompt
        cond_images = self._load_imgs_from_paths(img_path_list=img_cond_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)

        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}

        # prompt_embeds = self._encode_prompt(
        #     prompt,
        #     device,
        #     num_videos_per_prompt,
        #     prompt_embeds=prompt_embeds,
        # )



        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            ## process image embedding token
            if hasattr(self.pretrained_t2v, 'embedder'):
                uc_img = torch.zeros(batch_size, 3, height//self.vae_scale_factor, width//self.vae_scale_factor).to(self.pretrained_t2v.device)
                ## img: b c h w >> b l c
                uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None


        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps) # set_timesteps(num_inference_steps)   # , lcm_origin_steps)

        timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]

        # timesteps = self.scheduler.timesteps
        # timesteps[-1] = 0

        if skip_time_step_idx is not None:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")

        # self.scheduler.timesteps = timesteps

        print(f"[basic_sample] denoise timesteps: {timesteps}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable
        if latents is None:
            print("[basic_sample] latent is None, use full random init latent instead")
            assert (skip_time_step_idx is None) or (skip_time_step_idx == 0), "[basic_sample] skip time step should only work with prepared non full noise latents"
            num_channels_latents = unet_config["params"]["in_channels"]
            height = height // self.vae_scale_factor
            width = width // self.vae_scale_factor
            total_shape = (
                1,
                batch_size,
                num_channels_latents,
                frames,
                height,
                width,
            )
            print('total_shape', total_shape)
            latents = torch.randn(total_shape, device=device).repeat(1, batch_size, 1, 1, 1, 1)
            latents = latents[0]
        else:
            print("[basic_sample] using given init latent")

        bs = batch_size * num_videos_per_prompt # ?

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                kwargs.update({"clean_cond": True})

                ts = torch.full((bs,), t, device=device, dtype=torch.long)  # [1]

                model_pred_cond = self.pretrained_t2v.model( # self.unet(
                    latents,
                    ts,
                    **cond,
                    # timestep_cond=w_embedding.to(self.dtype),
                    curr_time_steps=ts,
                    temporal_length=frames,
                    **kwargs  # doubt: **kwargs?
                )

                if guidance_scale != 1.0:

                    model_pred_uncond = self.pretrained_t2v.model( # self.unet(
                        latents,
                        ts,
                        **uncond,
                        # timestep_cond=w_embedding.to(self.dtype),
                        curr_time_steps=ts,
                        temporal_length=frames,
                        **kwargs
                    )

                    model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)

                else:
                    model_pred = model_pred_cond

                # TODO : 增加 temporal guidance

                # result_dict = self.scheduler.step(
                #     model_output=model_pred,
                #     timestep=int(t),  # TODO : FIX
                #     sample=latents,
                #     return_dict=True
                # )

                # latents = result_dict.prev_sample
                # denoised = result_dict.pred_original_sample

                index = total_steps - i - 1     # Notes: 之前的 timesteps 进行了 flip, 所以要再倒转

                latents, denoised = self.scheduler.ddim_step(sample=latents, noise_pred=model_pred, indices=[index]*latents.shape[2])

                progress_bar.update()

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)
        else:
            videos = denoised

        return videos, denoised

    @torch.no_grad()
    def basic_sample_inpaint(
            self,
            prompt: Union[str, List[str]] = None,
            img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # TODO: 已给出部分的 noise 也应该保持与生产时使用的相同 ?

            given_clear_latent: torch.Tensor = None,   # 已有的 latent, 形状也为 320x512, 需要新生成的部分用 0 填充作为占位符(不会实际使用)
            inpaint_mask: torch.Tensor = None,         # given_latent 占据的部分为 1, inpaint 新生成的部分为 0, 需要与 given_clear_latent 相匹配

            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            skip_time_step_idx=None,
            **kwargs
    ):
        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if isinstance(img_cond_path, str):
                img_cond_path = [img_cond_path]
            else:
                assert len(img_cond_path) == 1, "[basic_sample] cond img should have same amount as text prompts"
        elif prompt is not None and isinstance(prompt, list):
            assert len(prompt) == len(img_cond_path), "[basic_sample] cond img should have same amount as text prompts"
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        cond_images = self._load_imgs_from_paths(img_path_list=img_cond_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)

        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}


        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            ## process image embedding token
            if hasattr(self.pretrained_t2v, 'embedder'):
                uc_img = torch.zeros(batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor).to(
                    self.pretrained_t2v.device)
                ## img: b c h w >> b l c
                uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)  # set_timesteps(num_inference_steps)   # , lcm_origin_steps)

        timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]


        if skip_time_step_idx is not None:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")


        print(f"[basic_sample] denoise timesteps: {timesteps}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable
        if latents is None:
            print("[basic_sample] latent is None, use full random init latent instead")
            assert (skip_time_step_idx is None) or (
                        skip_time_step_idx == 0), "[basic_sample] skip time step should only work with prepared non full noise latents"
            num_channels_latents = unet_config["params"]["in_channels"]
            height = height // self.vae_scale_factor
            width = width // self.vae_scale_factor
            total_shape = (
                1,
                batch_size,
                num_channels_latents,
                frames,
                height,
                width,
            )
            print('total_shape', total_shape)
            latents = torch.randn(total_shape, device=device).repeat(1, batch_size, 1, 1, 1, 1)
            latents = latents[0]
        else:
            print("[basic_sample] using given init latent")

        bs = batch_size * num_videos_per_prompt  # ?

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                kwargs.update({"clean_cond": True})

                ts = torch.full((bs,), t, device=device, dtype=torch.long)  # [1]

                model_pred_cond = self.pretrained_t2v.model(  # self.unet(
                    latents,
                    ts,
                    **cond,
                    # timestep_cond=w_embedding.to(self.dtype),
                    curr_time_steps=ts,
                    temporal_length=frames,
                    **kwargs  # doubt: **kwargs?
                )

                if guidance_scale != 1.0:

                    model_pred_uncond = self.pretrained_t2v.model(  # self.unet(
                        latents,
                        ts,
                        **uncond,
                        # timestep_cond=w_embedding.to(self.dtype),
                        curr_time_steps=ts,
                        temporal_length=frames,
                        **kwargs
                    )

                    model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)

                else:
                    model_pred = model_pred_cond

                index = total_steps - i - 1  # Notes: 之前的 timesteps 进行了 flip, 所以要再倒转

                latents, denoised = self.scheduler.ddim_step(sample=latents, noise_pred=model_pred,
                                                             indices=[index] * latents.shape[2])

                # inpaint
                noised_given_latent = self._add_noise(clear_video_latent=given_clear_latent, time_step_index=total_steps - i - 1)
                latents = torch.where(inpaint_mask == 1, noised_given_latent, latents)

                progress_bar.update()

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)
        else:
            videos = denoised

        return videos, denoised




    @torch.no_grad()
    def basic_sample_shift_multi_windows(
            self,
            prompt: Union[str, List[str]] = None,
            img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # TODO: 已给出部分的 noise 也应该保持与生产时使用的相同 ?

            init_panorama_latent: torch.Tensor = None,  # 包含整个 panorama 的 latent
            num_windows_w: int = None,                  # 总宽度是 width  (512) * num_windows_w
            num_windows_h: int = None,                  # 总高度是 height (320) * num_windows_h
            num_windows_f: int = None,                  # 总帧数是 frames (16)  * num_windows_f
            loop_step: int = None,                      # 应大于1, 越小 window 滑动越快      # doubt: 是否应该也分 h / w
            pano_image_path: str = None,

            dock_at_h=None,  # 是否多使用两组在上下边缘停靠的 window, 用于缓解上下越界融合问题

            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            merge_renoised_overlap_latent_ratio: float = 1,

            use_skip_time=False,
            skip_time_step_idx=None,
            progressive_skip=False,
            **kwargs
    ):
        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if isinstance(img_cond_path, str):
                img_cond_path = [img_cond_path]
            else:
                assert len(img_cond_path) == 1, "[basic_sample] cond img should have same amount as text prompts"
        elif prompt is not None and isinstance(prompt, list):
            assert len(prompt) == len(img_cond_path), "[basic_sample] cond img should have same amount as text prompts"
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        cond_images = self._load_imgs_from_paths(img_path_list=img_cond_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)

        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}


        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            ## process image embedding token
            if hasattr(self.pretrained_t2v, 'embedder'):
                uc_img = torch.zeros(batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor).to(
                    self.pretrained_t2v.device)
                ## img: b c h w >> b l c
                uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)  # set_timesteps(num_inference_steps)   # , lcm_origin_steps)

        timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]


        if use_skip_time and not progressive_skip:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")


        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable [pano]
        num_channels_latents = unet_config["params"]["in_channels"]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        total_shape = (
            batch_size,
            num_channels_latents,
            frames * num_windows_f,
            latent_height * num_windows_h,
            latent_width * num_windows_w,
        )

        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

            if use_skip_time:

                frame_0_latent = encode_images_list_to_latent_tensor(pretrained_t2v=self.pretrained_t2v,
                                                                     image_folder=None,
                                                                     image_size=(height * num_windows_h, width * num_windows_w),
                                                                     image_path_list=[pano_image_path])
                if progressive_skip:
                    for frame_idx, progs_skip_idx in enumerate(list(reversed(range(skip_time_step_idx)))):
                        # noised_frame_latent = self._add_noise(clear_video_latent=frame_0_latent,
                        #                                       time_step_index=total_steps-progs_skip_idx-1)
                        noised_frame_latent = self.scheduler.re_noise(x_a=frame_0_latent,
                                                                      step_a=0,
                                                                      step_b=total_steps - progs_skip_idx - 1)
                        init_panorama_latent[:, :, [frame_idx]] = noised_frame_latent.clone()

                else:
                    clear_repeat_latent = torch.cat([frame_0_latent] * frames * num_windows_f, dim=2)
                    # init_panorama_latent = self._add_noise(clear_video_latent=clear_repeat_latent,
                    #                                        time_step_index=total_steps-1)
                    init_panorama_latent = self.scheduler.re_noise(x_a=clear_repeat_latent,
                                                                   step_a=0,
                                                                   step_b=total_steps-1)

        else:
            print("[basic_sample_shift_multi_windows] using given init latent")
            assert init_panorama_latent.shape == total_shape, f"[basic_sample_shift_multi_windows] " \
                                                              f"init_panorama_latent shape {init_panorama_latent.shape}" \
                                                              f"does not match" \
                                                              f"desired shape {total_shape}"
            init_panorama_latent = init_panorama_latent.clone()


        panorama_ring_latent_handler = RingLatent(init_latent=init_panorama_latent)
        panorama_ring_latent_denoised_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))

        # define window shift
        image_step_size_w = width // loop_step
        latent_step_size_w = image_step_size_w // self.vae_scale_factor
        if num_windows_w == 1:
            latent_step_size_w = 0

        image_step_size_h = height // loop_step
        latent_step_size_h = image_step_size_h // self.vae_scale_factor
        if num_windows_h == 1:
            latent_step_size_h = 0

        latent_step_size_f = frames // loop_step
        if num_windows_f == 1:          # 不拓展时不动, 避免成环同时只有一个 window, frame 太少对画面的影响
            latent_step_size_f = 0

        assert latent_step_size_f > 0 or num_windows_f == 1, f"[basic_sample_shift_multi_windows] loop_step {loop_step} " \
                                                             f"> frames {frames} while num_windows_f {num_windows_f} > 0"
        # TODO: 增加精确适配 step_size 为 小数时的 SW 位置分配
        #  loop_step = 100
        #  latent_step_size_f = 16 / loop_step
        #  prev = 1
        #  for i in range(48):
        #      xx = (i % loop_step) * latent_step_size_f
        #      if int(xx) == prev:
        #          pp = int(xx) + (int(xx)-prev) - 1
        #      else:
        #          pp = int(xx)
        #      prev = pp
        #      print(f"[{i}({i % loop_step})] {round(xx, 4)}: \t int -> {int(xx)}, round -> {round(xx)}, pp={pp}")

        # prepare window image cond
        total_width = width * num_windows_w
        total_height = height * num_windows_h
        panorama_ring_image_tensor_handler = RingImageTensor(image_path=pano_image_path, height=total_height, width=total_width)

        bs = batch_size * num_videos_per_prompt  # ?

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                image_pos_left_start = (i % loop_step) * image_step_size_w       # 第一个window的左边缘起点
                image_pos_top_start = (i % loop_step) * image_step_size_h        # 第一个window的上边缘起点

                latent_pos_left_start = (i % loop_step) * latent_step_size_w    # 注意区分 latent 和 image 的索引大小
                latent_pos_top_start = (i % loop_step) * latent_step_size_h

                latent_frames_begin = (i % loop_step) * latent_step_size_f

                print(f"\n"
                      f"i = {i}, t = {t}")

                # reset denoised mask record
                panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))


                for shift_f_idx in range(num_windows_f):

                    for shift_w_idx in range(num_windows_w):

                        shift_h_idies_list = list(range(num_windows_h))
                        if dock_at_h:
                            shift_h_idies_list = [-100] + [-101] + list(range(num_windows_h))   # Notes: 把 -101 (dock at down edge) 放在前面, 以减轻断层现象

                        for shift_h_idx in shift_h_idies_list:

                            window_image_left = image_pos_left_start + shift_w_idx * width
                            window_image_right = window_image_left + width
                            window_image_top = image_pos_top_start + shift_h_idx * height
                            window_image_down = window_image_top + height

                            window_latent_left = latent_pos_left_start + shift_w_idx * latent_width
                            window_latent_right = window_latent_left + latent_width
                            window_latent_top = latent_pos_top_start + shift_h_idx * latent_height
                            window_latent_down = window_latent_top + latent_height

                            window_latent_frame_begin = latent_frames_begin + shift_f_idx * frames
                            window_latent_frame_end = window_latent_frame_begin + frames

                            if dock_at_h:
                                if shift_h_idx == -100: # dock at up edge
                                    if i % loop_step == 0:
                                        print(f"i % loop_step = {i} % {loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_top = 0
                                    window_latent_down = window_latent_top + latent_height
                                    window_image_top = 0
                                    window_image_down = window_image_top + height

                                if shift_h_idx == -101: # dock at down edge
                                    if i % loop_step == 0:
                                        print(f"i % loop_step = {i} % {loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_top = height * num_windows_h // self.vae_scale_factor - latent_height
                                    window_latent_down = window_latent_top + latent_height
                                    window_image_top = height * num_windows_h - height
                                    window_image_down = window_image_top + height

                                if window_latent_down > height * num_windows_h // self.vae_scale_factor:
                                    print(f"window_latent_down = {window_latent_down} > down edge = {height * num_windows_h // self.vae_scale_factor}, skipped because docking H")
                                    continue


                            window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                                                                                           pos_right=window_latent_right,
                                                                                           pos_top=window_latent_top,
                                                                                           pos_down=window_latent_down,
                                                                                           frame_begin=window_latent_frame_begin,
                                                                                           frame_end=window_latent_frame_end)

                            img_emb = panorama_ring_image_tensor_handler.get_encoded_image_cond(pretrained_t2v=self.pretrained_t2v,
                                                                                                pos_left=window_image_left,
                                                                                                pos_right=window_image_right,
                                                                                                pos_top=window_image_top,
                                                                                                pos_down=window_image_down)

                            window_denoised_mask = panorama_ring_mask_handler.get_window_latent(pos_left=window_latent_left,
                                                                                                pos_right=window_latent_right,
                                                                                                pos_top=window_latent_top,
                                                                                                pos_down=window_latent_down,
                                                                                                frame_begin=window_latent_frame_begin,
                                                                                                frame_end=window_latent_frame_end)

                            if merge_renoised_overlap_latent_ratio is not None and i < total_steps - 1:

                                noised_window_latent = self.scheduler.re_noise(x_a=window_latent.clone(),
                                                                               step_a=total_steps - i - 1 - 1,
                                                                               step_b=total_steps - i - 1)
                                window_denoised_mask = window_denoised_mask[0, 0, [0]]
                                window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                                      latent_to_add=noised_window_latent,
                                                                      mask=window_denoised_mask,
                                                                      mix_ratio=merge_renoised_overlap_latent_ratio)



                            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
                            cond = {"c_crossattn": [imtext_cond], "fps": fps}

                            print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
                                  f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
                                  f"window image: h[{window_image_top} - {window_image_down}] w[{window_image_left} - {window_image_right}]")

                            kwargs.update({"clean_cond": True})

                            ts = torch.full((bs,), t, device=device, dtype=torch.long)  # [1]

                            model_pred_cond = self.pretrained_t2v.model(  # self.unet(
                                window_latent,
                                ts,
                                **cond,
                                # timestep_cond=w_embedding.to(self.dtype),
                                curr_time_steps=ts,
                                temporal_length=frames,
                                **kwargs  # doubt: **kwargs?
                            )

                            if guidance_scale != 1.0:

                                model_pred_uncond = self.pretrained_t2v.model(  # self.unet(
                                    window_latent,
                                    ts,
                                    **uncond,
                                    # timestep_cond=w_embedding.to(self.dtype),
                                    curr_time_steps=ts,
                                    temporal_length=frames,
                                    **kwargs
                                )

                                model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)

                            else:
                                model_pred = model_pred_cond

                            index = total_steps - i - 1  # Notes: 之前的 timesteps 进行了 flip, 所以要再倒转

                            window_latent, denoised = self.scheduler.ddim_step(sample=window_latent, noise_pred=model_pred,
                                                                               indices=[index] * window_latent.shape[2])

                            panorama_ring_latent_handler.set_window_latent(window_latent,
                                                                           pos_left=window_latent_left,
                                                                           pos_right=window_latent_right,
                                                                           pos_top=window_latent_top,
                                                                           pos_down=window_latent_down,
                                                                           frame_begin=window_latent_frame_begin,
                                                                           frame_end=window_latent_frame_end)

                            panorama_ring_latent_denoised_handler.set_window_latent(denoised,
                                                                                    pos_left=window_latent_left,
                                                                                    pos_right=window_latent_right,
                                                                                    pos_top=window_latent_top,
                                                                                    pos_down=window_latent_down,
                                                                                    frame_begin=window_latent_frame_begin,
                                                                                    frame_end=window_latent_frame_end)

                            new_window_denoised_mask = torch.ones_like(window_latent, dtype=window_latent.dtype, device=window_latent.device)
                            panorama_ring_mask_handler.set_window_latent(new_window_denoised_mask,
                                                                         pos_left=window_latent_left,
                                                                         pos_right=window_latent_right,
                                                                         pos_top=window_latent_top,
                                                                         pos_down=window_latent_down,
                                                                         frame_begin=window_latent_frame_begin,
                                                                         frame_end=window_latent_frame_end)


                progress_bar.update()

        denoised = panorama_ring_latent_denoised_handler.torch_latent.clone().to(device=init_panorama_latent.device)

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)      # doubt: VAE 能正常 decode 超宽的 latent 吗 ?
        else:
            videos = denoised

        return videos, denoised



    @torch.no_grad()
    def fifo_sample(self,
                    total_video_length,
                    frame_length,
                    num_partitions,
                    lookahead_denoising,
                    num_inference_steps,

                    init_clear_latents,

                    prompt: Union[str, List[str]] = None,
                    img_cond_path: Union[str, List[str]] = None,        # TODO : 多图像支持 (FIFO中可以改为用即将 denoise 完毕出列的 frame 对应的原图来做当次的 condition?)
                    height: Optional[int] = 320,
                    width: Optional[int] = 512,
                    # frames: int = 16,
                    fps: int = 16,
                    guidance_scale: float = 7.5,
                    num_videos_per_prompt: Optional[int] = 1,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    # latents: Optional[torch.FloatTensor] = None,
                    # prompt_embeds: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    skip_time_step_idx=0,        # TODO : 支持 skip time step

                    save_frames=False,
                    output_dir=None,
                    **kwargs
                    ):

        if save_frames:
            assert output_dir is not None, "[fifo_sample] should specify an available path if save_frames is used"
            fifo_dir = os.path.join(output_dir, "fifo")
            os.makedirs(fifo_dir, exist_ok=True)


        # 2. Define call parameters
        assert prompt is not None, "[fifo_sample] prompt should not be none. prompt_embeds is not supported currently, (to be done)"
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if isinstance(img_cond_path, str):
                img_cond_path = [img_cond_path]
            else:
                assert len(img_cond_path) == 1, "[fifo_sample] cond img should have same amount as text prompts"
        elif isinstance(prompt, list):
            assert len(prompt) == len(img_cond_path), "[fifo_sample] cond img should have same amount as text prompts"
            batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # do_classifier_free_guidance = guidance_scale > 0.0  # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

        # 3. Encode input prompt
        cond_images = self._load_imgs_from_paths(img_path_list=img_cond_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)

        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}

        # prompt_embeds = self._encode_prompt(
        #     prompt,
        #     device,
        #     num_videos_per_prompt,
        #     prompt_embeds=prompt_embeds,
        # )

        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            ## process image embedding token
            if hasattr(self.pretrained_t2v, 'embedder'):
                uc_img = torch.zeros(batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor).to(
                    self.pretrained_t2v.device)
                ## img: b c h w >> b l c
                uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)
        timesteps = self.scheduler.ddim_timesteps        # [ 0, ... , 999 ]

        if skip_time_step_idx > 0:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")
            raise NotImplementedError()     # TODO : Work in process

        indices = np.arange(num_inference_steps - skip_time_step_idx)

        if lookahead_denoising:
            timesteps = np.concatenate([np.full((frame_length // 2,), timesteps[0]), timesteps])
            indices = np.concatenate([np.full((frame_length // 2,), 0), indices])

        # 5. Prepare init latents
        init_latents = self.prepare_latents(video_latent=init_clear_latents,
                                            lookahead_denoising=lookahead_denoising,
                                            num_inference_steps=num_inference_steps,
                                            frames_len=frame_length)

        frame_tensors_list = []
        fifo_video_frames = []
        moving_latents_queue = init_latents.clone()

        # FIFO Denoise loop
        # doubt : # for i in trange(new_video_length + num_inference_steps - frames, desc="fifo sampling"):
        for i in trange(total_video_length, desc="fifo sampling"):
            for rank in reversed(range(2 * num_partitions if lookahead_denoising else num_partitions)):
                start_idx = rank * ((frame_length - skip_time_step_idx) // 2) if lookahead_denoising else rank * (frame_length - skip_time_step_idx)
                midpoint_idx = start_idx + (frame_length - skip_time_step_idx) // 2
                end_idx = start_idx + (frame_length - skip_time_step_idx)

                idx = indices[start_idx:end_idx].copy()

                input_latents = moving_latents_queue[:, :, start_idx:end_idx].clone()
                ts = torch.Tensor(timesteps[start_idx:end_idx]).to(device=device, dtype=torch.long)
                print(f"denoising {ts}")
                # model_pred = self.unet(
                #     input_latents,
                #     ts,
                #     **context,
                #     timestep_cond=w_embedding.to(self.dtype),
                #     curr_time_steps=ts,
                #     idx_list=idx_list,
                #     input_traj=input_traj,
                #     temporal_length=frames-skip_time_step_idx,
                #     input_paths=current_paths,
                #     use_freetraj=attn_use_freetraj,
                #     use_free_traj_time_step_thres=use_free_traj_time_step_thres,
                #     **kwargs
                # )

                ts = torch.Tensor(timesteps[start_idx:end_idx]).to(device=device, dtype=torch.long)

                model_pred_cond = self.pretrained_t2v.model( # self.unet(
                    input_latents,
                    ts,
                    **cond,
                    # curr_time_steps=ts,
                    temporal_length=frame_length,
                    **kwargs  # doubt: **kwargs?
                )

                if guidance_scale != 1.0:
                    model_pred_uncond = self.pretrained_t2v.model( # self.unet(
                        input_latents,
                        ts,
                        **uncond,
                        # timestep_cond=w_embedding.to(self.dtype),
                        curr_time_steps=ts,
                        temporal_length=frame_length,
                        **kwargs
                    )
                    model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
                else:
                    model_pred = model_pred_cond

                output_latents, denoised = self.scheduler.ddim_step(
                    noise_pred=model_pred,
                    sample=input_latents,
                    indices=idx,
                )
                # output_latents, denoised = self.scheduler.fifo_step(
                #     model_pred=model_pred,
                #     timeindex_list=idx,
                #     timestep_list=ts,
                #     sample_latent=input_latents,
                #     return_dict=False
                # )

                if lookahead_denoising:
                    moving_latents_queue[:, :, midpoint_idx:end_idx] = output_latents[:, :, -((frame_length - skip_time_step_idx) // 2):]
                else:
                    moving_latents_queue[:, :, start_idx:end_idx] = output_latents
                del output_latents

            # reconstruct from latent to pixel space
            first_frame_idx = (frame_length - skip_time_step_idx) // 2 if lookahead_denoising else 0
            frame_tensor = self.pretrained_t2v.decode_first_stage_2DAE(moving_latents_queue[:, :, [first_frame_idx]])  # b,c,1,H,W
            frame_tensors_list.append(frame_tensor)
            image = self.tensor2image(frame_tensor)
            if save_frames:
                fifo_path = os.path.join(fifo_dir, f"{i}.png")
                image.save(fifo_path)
            fifo_video_frames.append(image)

            # shift FIFO latent queue
            moving_latents_queue = self.shift_latents(moving_latents=moving_latents_queue, enqueue_init_latents=None)

        frame_tensors = torch.cat(frame_tensors_list, dim=2)
        return fifo_video_frames, frame_tensors

    def shift_latents(self, moving_latents, enqueue_init_latents=None):
        """
        enqueue_init_latent: shape: [b, c, 1, h, w]
        """
        # shift latents
        moving_latents[:, :, :-1] = moving_latents[:, :, 1:].clone()

        if enqueue_init_latents is None:
            enqueue_init_latents = torch.randn_like(moving_latents[:, :, [0]]).to(device=moving_latents.device, dtype=moving_latents.dtype)

        # add new noise to the last frame
        moving_latents[:, :, [-1]] = enqueue_init_latents.clone()

        return moving_latents

    @torch.no_grad()
    def prepare_latents(self, video_latent, lookahead_denoising, num_inference_steps, frames_len):
        # assert video_latent.shape[2] == num_inference_steps
        latents_list = []
        ddim_alphas = self.scheduler.ddim_alphas

        if lookahead_denoising:
            for i in range(frames_len // 2):
                alpha = ddim_alphas[0]
                beta = 1 - alpha
                latents = (alpha ** 0.5) * video_latent[:, :, [0]] + (beta ** 0.5) * torch.randn_like(video_latent[:, :, [0]])
                latents_list.append(latents)

        for i in range(num_inference_steps):
            alpha = ddim_alphas[i]  # image -> noise
            beta = 1 - alpha
            # frame_idx = max(0, i - (num_inference_steps - frames_len))
            frame_idx = i   # TODO : fix frame_idx
            latents = (alpha ** 0.5) * video_latent[:, :, [frame_idx]] + (beta ** 0.5) * torch.randn_like(video_latent[:, :, [frame_idx]])
            latents_list.append(latents)

        latents = torch.cat(latents_list, dim=2)

        return latents

    @torch.no_grad()
    def _add_noise(self, clear_video_latent, time_step_index):  # TODO : 改成用 scheduler.re_noise ?
        ddim_alphas = self.scheduler.ddim_alphas
        alpha = ddim_alphas[time_step_index]
        beta = 1 - alpha
        noised_latent = (alpha ** 0.5) * clear_video_latent + (beta ** 0.5) * torch.randn_like(clear_video_latent)
        return noised_latent

    def tensor2image(self, batch_tensors):
        img_tensor = torch.squeeze(batch_tensors)  # c,h,w

        image = img_tensor.detach().cpu()
        image = torch.clamp(image.float(), -1., 1.)

        image = (image + 1.0) / 2.0
        image = (image * 255).to(torch.uint8).permute(1, 2, 0)  # h,w,c
        image = image.numpy()
        image = Image.fromarray(image)

        return image

    @torch.no_grad()
    def encode_image_cond(self, img_path, height, width):
        cond_images = self._load_imgs_from_paths(img_path_list=img_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        return img_emb

    @torch.no_grad()
    def fifo_loop_sample(self,
                    total_video_length,
                    frame_length,
                    num_partitions,
                    lookahead_denoising,
                    num_inference_steps,

                    init_clear_latents,

                    prompt: Union[str, List[str]] = None,
                    img_cond_path: Union[str, List[str]] = None,        # TODO : 多图像支持 (FIFO中可以改为用即将 denoise 完毕出列的 frame 对应的原图来做当次的 condition?)
                    height: Optional[int] = 320,
                    width: Optional[int] = 512,

                    use_moving_img_cond = False,
                    shift_moving_img_cond_by_rank=False,
                    img_path_list = None,

                    # frames: int = 16,
                    fps: int = 16,
                    guidance_scale: float = 7.5,
                    num_videos_per_prompt: Optional[int] = 1,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    # latents: Optional[torch.FloatTensor] = None,
                    # prompt_embeds: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    # skip_time_step_idx=0,        # TODO : 支持 skip time step


                    skip_num_partition=0,
                    prefix_sampling_step=0,
                    prefix_clear_latents=None,

                    save_frames=False,
                    output_dir=None,
                    **kwargs
                    ):

        if save_frames:
            assert output_dir is not None, "[fifo_loop_sample] should specify an available path if save_frames is used"
            fifo_dir = os.path.join(output_dir, "fifo")
            os.makedirs(fifo_dir, exist_ok=True)


        # 2. Define call parameters
        assert prompt is not None, "[fifo_loop_sample] prompt should not be none. prompt_embeds is not supported currently, (to be done)"
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if isinstance(img_cond_path, str):
                img_cond_path = [img_cond_path]
            else:
                assert len(img_cond_path) == 1, "[fifo_loop_sample] cond img should have same amount as text prompts"
        elif isinstance(prompt, list):
            assert len(prompt) == len(img_cond_path), "[fifo_loop_sample] cond img should have same amount as text prompts"
            batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # do_classifier_free_guidance = guidance_scale > 0.0  # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)


        # 3. Encode input prompt
        if use_moving_img_cond:
            assert img_path_list is not None, "[fifo_loop_sample] must specify a image path list to enable use_moving_img_cond "

            print("[fifo_loop_sample] use_moving_img_cond enabled, encoding each img_cond ...")
            # img_cond_path = [img_path_list[0]]

            moving_cond = []

            for img_path in img_path_list:
                img_emb = self.encode_image_cond(img_path=[img_path], height=height, width=width)
                text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)
                imtext_cond = torch.cat([text_emb, img_emb], dim=1)
                cond = {"c_crossattn": [imtext_cond], "fps": fps}
                moving_cond.append(cond)

        else:
            img_emb = self.encode_image_cond(img_path=img_cond_path, height=height, width=width)
            text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}

        # prompt_embeds = self._encode_prompt(
        #     prompt,
        #     device,
        #     num_videos_per_prompt,
        #     prompt_embeds=prompt_embeds,
        # )

        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            ## process image embedding token
            if hasattr(self.pretrained_t2v, 'embedder'):
                uc_img = torch.zeros(batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor).to(
                    self.pretrained_t2v.device)
                ## img: b c h w >> b l c
                uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)
        timesteps = self.scheduler.ddim_timesteps        # [ 0, ... , 999 ]

        assert skip_num_partition < num_partitions, f"can not skip more than num_partitions {num_partitions} "
        num_partition_skipped = num_partitions - skip_num_partition
        num_inference_steps_skipped = num_inference_steps - skip_num_partition * frame_length
        if skip_num_partition > 0:
            timesteps = timesteps[:-(skip_num_partition * frame_length)]
            print(f"skip : {skip_num_partition}*{frame_length} step")
            # raise NotImplementedError()     # TODO : Work in process


        indices = np.arange(num_inference_steps_skipped)

        if lookahead_denoising:
            timesteps = np.concatenate([np.full((frame_length // 2,), timesteps[0]), timesteps])
            indices = np.concatenate([np.full((frame_length // 2,), 0), indices])


        # 5. Prepare init latents
        if prefix_sampling_step > 0:
            # assert prefix_clear_latents.shape[2] == prefix_sampling_step == num_inference_steps, "prefix step should match latent len"
            print(f"prefix_clear_latents.shape[2]: {prefix_clear_latents.shape[2]} => num_inference_steps_skipped: {num_inference_steps_skipped}")
            init_latents = self.prepare_latents(video_latent=prefix_clear_latents,
                                                lookahead_denoising=lookahead_denoising,
                                                num_inference_steps=num_inference_steps_skipped,
                                                frames_len=frame_length)
        else:
            init_latents = self.prepare_latents(video_latent=init_clear_latents,
                                                lookahead_denoising=lookahead_denoising,
                                                num_inference_steps=num_inference_steps_skipped,
                                                frames_len=frame_length)

        frame_tensors_list = []
        fifo_video_frames = []
        moving_latents_queue = init_latents.clone()

        # FIFO Denoise loop
        for i in trange(prefix_sampling_step + total_video_length, desc="fifo sampling"):
        # doubt : # for i in trange(total_video_length + num_inference_steps - frame_length, desc="fifo sampling"):

            if use_moving_img_cond:
                print("")   # 对齐输出

            for rank in reversed(range(2 * num_partition_skipped if lookahead_denoising else num_partition_skipped)):
                start_idx = rank * (frame_length // 2) if lookahead_denoising else rank * frame_length
                midpoint_idx = start_idx + frame_length // 2
                end_idx = start_idx + frame_length

                idx = indices[start_idx:end_idx].copy()

                input_latents = moving_latents_queue[:, :, start_idx:end_idx].clone()
                ts = torch.Tensor(timesteps[start_idx:end_idx]).to(device=device, dtype=torch.long)

                # print(f"denoising {ts}")
                # model_pred = self.unet(
                #     input_latents,
                #     ts,
                #     **context,
                #     timestep_cond=w_embedding.to(self.dtype),
                #     curr_time_steps=ts,
                #     idx_list=idx_list,
                #     input_traj=input_traj,
                #     temporal_length=frames-skip_time_step_idx,
                #     input_paths=current_paths,
                #     use_freetraj=attn_use_freetraj,
                #     use_free_traj_time_step_thres=use_free_traj_time_step_thres,
                #     **kwargs
                # )

                if use_moving_img_cond:
                    if prefix_sampling_step > 0:
                        if shift_moving_img_cond_by_rank:
                            img_cond_idx = min(max(0, i + start_idx - prefix_sampling_step), len(moving_cond)-1)
                        else:
                            img_cond_idx = max(0, i - prefix_sampling_step)
                        cond = moving_cond[img_cond_idx]
                        print(f"idx: {idx} : {[max(0, index + i - prefix_sampling_step) for index in idx]}"
                              f" => moving_cond: [{img_cond_idx}]")
                    else:
                        cond = moving_cond[i]

                model_pred_cond = self.pretrained_t2v.model( # self.unet(
                    input_latents,
                    ts,
                    **cond,
                    # curr_time_steps=ts,
                    temporal_length=frame_length,
                    **kwargs  # doubt: **kwargs?
                )

                if guidance_scale != 1.0:
                    model_pred_uncond = self.pretrained_t2v.model( # self.unet(
                        input_latents,
                        ts,
                        **uncond,
                        # timestep_cond=w_embedding.to(self.dtype),
                        curr_time_steps=ts,
                        temporal_length=frame_length,
                        **kwargs
                    )
                    model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
                else:
                    model_pred = model_pred_cond

                output_latents, denoised = self.scheduler.ddim_step(
                    noise_pred=model_pred,
                    sample=input_latents,
                    indices=idx,
                )
                # output_latents, denoised = self.scheduler.fifo_step(
                #     model_pred=model_pred,
                #     timeindex_list=idx,
                #     timestep_list=ts,
                #     sample_latent=input_latents,
                #     return_dict=False
                # )

                if lookahead_denoising:
                    moving_latents_queue[:, :, midpoint_idx:end_idx] = output_latents[:, :, -(frame_length // 2):]
                else:
                    moving_latents_queue[:, :, start_idx:end_idx] = output_latents
                del output_latents

            # reconstruct from latent to pixel space
            first_frame_idx = frame_length // 2 if lookahead_denoising else 0
            frame_tensor = self.pretrained_t2v.decode_first_stage_2DAE(moving_latents_queue[:, :, [first_frame_idx]])  # b,c,1,H,W
            frame_tensors_list.append(frame_tensor)
            image = self.tensor2image(frame_tensor)
            if save_frames:
                fifo_path = os.path.join(fifo_dir, f"{i}.png")
                image.save(fifo_path)
            fifo_video_frames.append(image)

            # loop
            # if prefix_sampling_step > 0:
            #     next_enqueue_index = max(0, min(total_video_length - 1, i + num_inference_steps_skipped - prefix_sampling_step))
            #     print(f"next enqueue : init_clear_latents[:, :, [{next_enqueue_index}]]")
            # else:
            #     # doubt: check
            #     next_enqueue_index = max(0, min(total_video_length - 1, i + num_inference_steps_skipped - prefix_sampling_step))
            #     # min(num_inference_steps_skipped - 1, i + num_inference_steps_skipped)
            #     # (i+num_inference_steps_skipped) % num_inference_steps

            # determine next enqueue frame latent index # TODO: CHECK
            next_enqueue_index = i + num_inference_steps_skipped - prefix_sampling_step
            if next_enqueue_index < 0:
                next_enqueue_index = 0
            elif next_enqueue_index >= init_clear_latents.shape[2]:
                next_enqueue_index = init_clear_latents.shape[2] - 1

            enqueue_init_latents = self._add_noise(clear_video_latent=init_clear_latents[:, :, [next_enqueue_index]].clone(), time_step_index=num_inference_steps_skipped-1)    # doubt: 这里是不是应该改为 total - num_inference_steps_skipped
            # shift FIFO latent queue
            moving_latents_queue = self.shift_latents(moving_latents=moving_latents_queue, enqueue_init_latents=enqueue_init_latents)

        frame_tensors = torch.cat(frame_tensors_list, dim=2)
        return fifo_video_frames, frame_tensors
