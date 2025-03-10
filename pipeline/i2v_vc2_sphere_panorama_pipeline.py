import warnings

import math
import numpy as np
import torch
from PIL import Image

from tqdm import trange
from typing import List, Optional, Union, Dict, Any
from diffusers import logging

from diffusers import DiffusionPipeline
from lvdm.models.ddpm3d import LatentVisualDiffusion

from pipeline.vc2_lvdm_scheduler import lvdm_DDIM_Scheduler
from pipeline.i2v_vc2_pipeline import VC2_Pipeline_I2V
from utils.diffusion_utils import resize_video_latent
from utils.multi_prompt_utils import select_prompt_from_multi_prompt_dict_by_factor
from utils.ring_panorama_tensor_utils import RingPanoramaLatentProxy
from utils.shift_window_utils import RingLatent, RingImageTensor
from utils.tensor_utils import load_image_tensor_from_path, mix_latents_with_mask

from utils.precast_latent_utils import encode_images_list_to_latent_tensor, _load_and_preprocess_image
from utils.panorama_tensor_utils import PanoramaTensor, PanoramaLatentProxy


class VC2_Pipeline_I2V_SpherePano(VC2_Pipeline_I2V):


    @torch.no_grad()
    def basic_sample_shift_shpere_panorama(
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

            init_sphere_latent: torch.Tensor = None,  # 包含整个 panorama 的 latent
            # num_windows_w: int = None,                  # 总宽度是 width  (512) * num_windows_w
            # num_windows_h: int = None,                  # 总高度是 height (320) * num_windows_h
            # num_windows_f: int = None,                  # 总帧数是 frames (16)  * num_windows_f
            # loop_step: int = None,                      # 应大于1, 越小 window 滑动越快

            pano_image_path: str = None,

            total_f: int = None,
            dock_at_f = None,
            overlap_ratio_list_f: List[float] = None,
            loop_step_frame: int = None,

            equirect_width: int = None,
            equirect_height: int = None,

            phi_theta_dict: dict = None,            # key: 俯仰角 phi (int, 0~360) -> value: 航向角 theta (int, 0~360)
            phi_prompt_dict: dict = None,           # key: 俯仰角 phi -> value: 对应的 prompt (str)

            view_fov: int = None,

            view_get_scale_factor: int = 1,         # 在 PanoramaTensor get 时请求比 latent 大多少倍的 view latent
            view_set_scale_factor: int = 1,         # 可能可以降低 Sphere Panorama 映射误差

            loop_step_theta: int = None,            # 多少步转满一个 window (view_fov)

            merge_renoised_overlap_latent_ratio: float = None,  # 将 window 间 overlap (已经被其他 window 去噪过) 的部分
                                                                # 重新加噪再融合到当前 view 的 latent 中进行去噪
                                                                # 值越大代表 renoise 部分比重越大
            merge_prev_denoised_ratio_list: List[float] = None,

            denoise_to_step: int = None,            # 去噪到多少步 (用于切换到一般平面去噪模式)
            paste_on_static = None,                 # 是否在映射写回 sphere panorama 时写回到静态图复制成的 pano tensor 中

            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            downsample_factor_before_vae_decode=None,   # 为了防止 latent 过大, 先降采样再 进行 VAE decode

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

        if denoise_to_step is not None:
            assert not (use_skip_time and not progressive_skip), 'should not use denoise_to_step while using Non progressive time step skip. ' \
                                                                 'which may lead to disordered time step count in following steps'
            if use_skip_time and not progressive_skip:
                assert skip_time_step_idx < denoise_to_step, f'denoise_to_step {denoise_to_step} should not stop denoise earlier than progressive skip {skip_time_step_idx}, ' \
                                                             'you may have set undesired values'

            timesteps = timesteps[:denoise_to_step]
            print(f"Denoised to : {denoise_to_step}")

        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        if use_skip_time:
            print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable [pano]
        num_channels_latents = unet_config["params"]["in_channels"]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        if total_f is None:
            total_f = frames

        sphere_latent_shape = (
            batch_size,
            num_channels_latents,
            frames if total_f is None else total_f, # * num_windows_f,
            equirect_height // self.vae_scale_factor, # latent_height * num_windows_h,
            equirect_width // self.vae_scale_factor, # latent_width * num_windows_w,
        )

        if init_sphere_latent is None:

            init_sphere_latent = torch.randn(sphere_latent_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

            if use_skip_time:

                # raise NotImplementedError       # TODO

                # frame_0_latent = encode_images_list_to_latent_tensor(pretrained_t2v=self.pretrained_t2v,
                #                                                      image_folder=None,
                #                                                      image_size=(equirect_height, equirect_width),
                #                                                      image_path_list=[pano_image_path])
                frame_0_latent = self.tiled_vae_encode_image(image_path=pano_image_path,
                                                             image_size=(equirect_height, equirect_width))

                if progressive_skip:
                    for frame_idx, progs_skip_idx in enumerate(list(reversed(range(skip_time_step_idx)))):

                        noised_frame_latent = self.scheduler.re_noise(x_a=frame_0_latent,
                                                                      step_a=0,
                                                                      step_b=total_steps - progs_skip_idx - 1)
                        init_sphere_latent[:, :, [frame_idx]] = noised_frame_latent.clone()
                        print(f"progressive_skip: frame[{frame_idx}] will have noise level {total_steps - progs_skip_idx - 1}")

                else:
                    # clear_repeat_latent = torch.cat([frame_0_latent] * frames * num_windows_f, dim=2)
                    if total_f is None:
                        clear_repeat_latent = torch.cat([frame_0_latent] * frames * 1, dim=2)
                    else:
                        clear_repeat_latent = torch.cat([frame_0_latent] * total_f * 1, dim=2)
                    init_sphere_latent = self.scheduler.re_noise(x_a=clear_repeat_latent,
                                                                 step_a=0,
                                                                 step_b=total_steps-1)

        else:
            print("[basic_sample_shift_multi_windows] using given init latent")
            assert init_sphere_latent.shape == sphere_latent_shape, f"[basic_sample_shift_multi_windows] " \
                                                                      f"init_panorama_latent shape {init_sphere_latent.shape}" \
                                                                      f"does not match" \
                                                                      f"desired shape {init_sphere_latent}"
            init_sphere_latent = init_sphere_latent.clone()


        panorama_sphere_latent_handler = RingPanoramaLatentProxy(equirect_tensor=init_sphere_latent)
        panorama_sphere_denoised_latent_handler = RingPanoramaLatentProxy(equirect_tensor=torch.zeros_like(init_sphere_latent))
        panorama_sphere_denoised_mask_handler = RingPanoramaLatentProxy(equirect_tensor=torch.zeros_like(init_sphere_latent))

        # prepare window image cond
        condition_image_tensor = load_image_tensor_from_path(image_path=pano_image_path, height=equirect_height, width=equirect_width)
        panorama_sphere_image_tensor_handler = PanoramaTensor(equirect_tensor=condition_image_tensor)


        # if phi_prompt_dict is not None:
        #     raise NotImplementedError       # TODO: 实现不同俯仰角使用不同 prompt

        bs = batch_size * num_videos_per_prompt  # ?

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:

            for i, t in enumerate(timesteps):

                phi_offset = 0      # doubt: 是否需要支持 phi 的 shift ?
                theta_offset = (i % loop_step_theta) * (view_fov // loop_step_theta)

                print(f"\n"
                      f"i = {i}, t = {t}")
                print(f"theta offset = {theta_offset}, phi_offset = {phi_offset}")

                # reset mask record
                panorama_sphere_denoised_mask_handler = RingPanoramaLatentProxy(equirect_tensor=torch.zeros_like(init_sphere_latent))


                if paste_on_static and i < total_steps - 1:

                    # frame_0_latent = encode_images_list_to_latent_tensor(pretrained_t2v=self.pretrained_t2v,
                    #                                                      image_folder=None,
                    #                                                      image_size=(equirect_height, equirect_width),
                    #                                                      image_path_list=[pano_image_path])
                    frame_0_latent = self.tiled_vae_encode_image(image_path=pano_image_path,
                                                                 image_size=(equirect_height, equirect_width))

                    clear_repeat_latent = torch.cat([frame_0_latent] * total_f, dim=2)
                    noised_repeat_latent = self.scheduler.re_noise(x_a=clear_repeat_latent,
                                                                   step_a=0,
                                                                   step_b=total_steps - i - 1)
                    temp_panorama_sphere_latent_handler = RingPanoramaLatentProxy(equirect_tensor=noised_repeat_latent)



                # Long Video

                _DOCK_START_INDEX = -101
                _DOCK_END_INDEX = -111

                overlap_ratio_f = overlap_ratio_list_f[i]
                total_window_num_f = math.ceil((total_f // frames - 1) / (1 - overlap_ratio_f)) + 1

                if total_f > frames:    # 需要 shift on F

                    offset_shift_step_size_f = max(int(overlap_ratio_f * frames / loop_step_frame), 1)
                    # TODO: check, 这里为了测试强行设置了最小值为 1
                    latent_frames_begin = (i % loop_step_frame) * offset_shift_step_size_f
                    curr_shift_f_idies_list = list(range(total_window_num_f))

                    if dock_at_f:
                        curr_shift_f_idies_list = [_DOCK_START_INDEX] + curr_shift_f_idies_list + [_DOCK_END_INDEX]

                elif total_f == frames: # 无需 shift on F
                    latent_frames_begin = 0
                    curr_shift_f_idies_list = [0]
                else:
                    print(f"total_f {total_f} should >= frames {frames} !")
                    raise ValueError

                for shift_f_idx in curr_shift_f_idies_list:
                    window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio_f))
                    window_latent_frame_begin = window_latent_frame_begin % total_f
                    window_latent_frame_end = window_latent_frame_begin + frames

                    if shift_f_idx == -777:
                        if i == 0:  # Notes: 在 i = 0 时 window 是恰好对齐 frames 数量的, 故增加一个平滑过渡机制,
                            #   也可以考虑改为算 latent_frames_begin 时 i 增加一个 offset
                            #   TODO: 增多这种过长"回头"去噪的frame长度
                            shift_f_idx = total_window_num_f
                            window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(
                                frames * (1 - overlap_ratio_f))
                            window_latent_frame_end = window_latent_frame_begin + frames
                        else:
                            continue

                    if dock_at_f:
                        if shift_f_idx == _DOCK_START_INDEX:
                            if latent_frames_begin == 0:
                                print(
                                    f"i % init_stage_loop_step = {i} % {loop_step_frame} = 0, no need for docking, skipped"
                                )
                                continue
                            window_latent_frame_begin = 0
                            window_latent_frame_end = window_latent_frame_begin + frames

                        if shift_f_idx == _DOCK_END_INDEX:
                            if latent_frames_begin == 0:
                                print(
                                    f"i % init_stage_loop_step = {i} % {loop_step_frame} = 0, no need for docking, skipped"
                                )
                                continue
                            window_latent_frame_begin = total_f - frames
                            window_latent_frame_end = window_latent_frame_begin + frames

                        if window_latent_frame_end > total_f:
                            print(
                                f"window_latent_frame_end = {window_latent_frame_end} > frame end edge = {total_f}, skipped because docking F")
                            continue

                    print(f"window_idx: frame [{shift_f_idx}] | \t "
                          f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] \t")

                    for phi_angle in list(phi_theta_dict.keys()):

                        theta_angles = phi_theta_dict[phi_angle]

                        # random.shuffle(theta_angles)

                        for theta_angle in theta_angles:

                            curr_phi_angle = phi_angle + phi_offset
                            curr_theta_angle = theta_angle + theta_offset

                            # if paste_on_static and i < total_steps - 1:
                            #     view_latent, _ = temp_panorama_sphere_latent_handler.get_view_tensor_no_interpolate(fov=view_fov,
                            #                                                                                         theta=curr_theta_angle,
                            #                                                                                         phi=curr_phi_angle,
                            #                                                                                         width=latent_width * view_get_scale_factor,
                            #                                                                                         height=latent_height * view_get_scale_factor)

                            view_latent, _ = panorama_sphere_latent_handler.get_view_tensor_no_interpolate(fov=view_fov,
                                                                                                           theta=curr_theta_angle,
                                                                                                           phi=curr_phi_angle,
                                                                                                           width=latent_width * view_get_scale_factor,
                                                                                                           height=latent_height * view_get_scale_factor,
                                                                                                           frame_begin=window_latent_frame_begin,
                                                                                                           frame_end=window_latent_frame_end)

                            if view_get_scale_factor != 1:
                                view_latent = resize_video_latent(input_latent=view_latent, mode="nearest",
                                                                  target_width=latent_width,
                                                                  target_height=latent_height)

                            view_latent_prev_denoise = view_latent.clone()

                            view_denoised_mask, _ = panorama_sphere_denoised_mask_handler.get_view_tensor_no_interpolate(
                                fov=view_fov,
                                theta=curr_theta_angle,
                                phi=curr_phi_angle,
                                width=latent_width,
                                height=latent_height,
                                frame_begin=window_latent_frame_begin,
                                frame_end=window_latent_frame_end,
                            )   # [1, H, W], 0 为没去噪过的部分, 1为已经去噪过的部分
                            # doubt: mask 是否需要应用 view_get_scale_factor ?

                            if merge_renoised_overlap_latent_ratio is not None and i < total_steps - 1 :

                                # noise_to_add = torch.zeros_like(view_latent, dtype=view_latent.dtype, device=view_latent.device)
                                noised_view_latent = self.scheduler.re_noise(x_a=view_latent.clone(),
                                                                             step_a=total_steps - i - 1 - 1,
                                                                             step_b=total_steps - i - 1)

                                view_latent = mix_latents_with_mask(latent_1=view_latent,
                                                                    latent_to_add=noised_view_latent,
                                                                    mask=view_denoised_mask,
                                                                    mix_ratio=merge_renoised_overlap_latent_ratio)

                            view_condition_image_tensor, _ = panorama_sphere_image_tensor_handler.get_view_tensor_no_interpolate(
                                fov=view_fov,
                                theta=curr_theta_angle,
                                phi=curr_phi_angle,
                                width=width,
                                height=height)

                            view_condition_image_tensor = view_condition_image_tensor.to(self.pretrained_t2v.device).unsqueeze(dim=0)  # `get_image_embeds` expects [b, c, h, w]

                            img_emb = self.pretrained_t2v.get_image_embeds(batch_imgs=view_condition_image_tensor)

                            print_prompt = prompt
                            if phi_prompt_dict is not None:
                                curr_phi_prompt = phi_prompt_dict[phi_angle]
                                text_emb = self.pretrained_t2v.get_learned_conditioning([curr_phi_prompt])
                                print_prompt = curr_phi_prompt

                            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
                            cond = {"c_crossattn": [imtext_cond], "fps": fps}

                            print(f"window: phi = {curr_phi_angle}, theta = {curr_theta_angle}, prompt = {print_prompt}")

                            kwargs.update({"clean_cond": True})

                            ts = torch.full((bs,), t, device=device, dtype=torch.long)  # [1]

                            model_pred_cond = self.pretrained_t2v.model(  # self.unet(
                                view_latent,
                                ts,
                                **cond,
                                # timestep_cond=w_embedding.to(self.dtype),
                                curr_time_steps=ts,
                                temporal_length=frames,
                                **kwargs  # doubt: **kwargs?
                            )

                            if guidance_scale != 1.0:

                                model_pred_uncond = self.pretrained_t2v.model(  # self.unet(
                                    view_latent,
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

                            view_latent, denoised = self.scheduler.ddim_step(sample=view_latent, noise_pred=model_pred,
                                                                             indices=[index] * view_latent.shape[2])

                            if view_set_scale_factor != 1:
                                view_latent = resize_video_latent(input_latent=view_latent, mode="nearest",
                                                                  target_width=latent_width * view_set_scale_factor,
                                                                  target_height=latent_height * view_set_scale_factor)
                                denoised = resize_video_latent(input_latent=denoised, mode="nearest",
                                                               target_width=latent_width * view_set_scale_factor,
                                                               target_height=latent_height * view_set_scale_factor)

                            if merge_prev_denoised_ratio_list is not None and i < total_steps - 1:     # Notes: 加权平均 overlap 部分的去噪结果
                                merge_prev_denoised_ratio = merge_prev_denoised_ratio_list[i]
                                view_latent = mix_latents_with_mask(latent_1=view_latent,
                                                                    latent_to_add=view_latent_prev_denoise,
                                                                    mask=view_denoised_mask,
                                                                    mix_ratio=merge_prev_denoised_ratio)

                            panorama_sphere_latent_handler.set_view_tensor_no_interpolation(
                                view_tensor=view_latent,
                                fov=view_fov,
                                theta=curr_theta_angle,
                                phi=curr_phi_angle,
                                frame_begin=window_latent_frame_begin,
                                frame_end=window_latent_frame_end,
                            )

                            if paste_on_static and i < total_steps - 1:
                                temp_panorama_sphere_latent_handler.set_view_tensor_no_interpolation(
                                    view_tensor=view_latent,
                                    fov=view_fov,
                                    theta=curr_theta_angle,
                                    phi=curr_phi_angle,
                                    frame_begin=window_latent_frame_begin,
                                    frame_end=window_latent_frame_end,
                                )

                            panorama_sphere_denoised_latent_handler.set_view_tensor_no_interpolation(view_tensor=denoised,
                                                                                                     fov=view_fov,
                                                                                                     theta=curr_theta_angle,
                                                                                                     phi=curr_phi_angle,
                                                                                                     frame_begin=window_latent_frame_begin,
                                                                                                     frame_end=window_latent_frame_end,
                                                                                                     )

                            new_view_denoised_mask = torch.ones_like(view_latent, dtype=view_latent.dtype, device=view_latent.device)
                            panorama_sphere_denoised_mask_handler.set_view_tensor_no_interpolation(view_tensor=new_view_denoised_mask,
                                                                                                   fov=view_fov,
                                                                                                   theta=curr_theta_angle,
                                                                                                   phi=curr_phi_angle,
                                                                                                   frame_begin=window_latent_frame_begin,
                                                                                                   frame_end=window_latent_frame_end,
                                                                                                   )

                if paste_on_static and i < total_steps - 1:
                    panorama_sphere_latent_handler = RingPanoramaLatentProxy(temp_panorama_sphere_latent_handler.get_equirect_tensor())

                progress_bar.update()

        denoised = panorama_sphere_denoised_latent_handler.get_equirect_tensor().to(device=self.pretrained_t2v.device)
        final_latents = panorama_sphere_latent_handler.get_equirect_tensor().to(device=self.pretrained_t2v.device)

        if downsample_factor_before_vae_decode is not None:
            B, C, N, H, W = denoised.shape
            denoised = resize_video_latent(input_latent=denoised.clone(), mode="nearest",
                                           target_height=int(H // downsample_factor_before_vae_decode),
                                           target_width=int(W // downsample_factor_before_vae_decode))
            final_latents = resize_video_latent(input_latent=final_latents.clone(), mode="nearest",
                                                target_height=int(H // downsample_factor_before_vae_decode),
                                                target_width=int(W // downsample_factor_before_vae_decode))

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)      # doubt: VAE 能正常 decode 超宽的 latent 吗 ?
        else:
            videos = final_latents

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
            total_w: int = None,
            total_h: int = None,                        # 允许任意自定义宽高, window 间可以重叠
            total_f: int = None,
            num_windows_w: int = None,                  # W 方向 window 数
            num_windows_h: int = None,
            num_windows_f: int = None,                  # 总帧数是 frames (16)  * num_windows_f
            loop_step: int = None,                      # 应大于1, 越小 window 滑动越快      # doubt: 是否应该也分 h / w

            begin_index_offset: int = 0,                # 计算每个 timestep 对应的H,W,F起始点前加入 offset,
                                                        # 可能缓解边缘接缝 gap 问题

            dock_at_f=None,
            overlap_ratio_list_f: List[float] = None,
            loop_step_frame: int = None,

            pano_image_path: str = None,

            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            merge_renoised_overlap_latent_ratio:float = 1,
            merge_prev_denoised_ratio_list: List[float] = None,

            window_multi_prompt_dict: dict = None,

            tiled_decode = True, # 使用分块 VAE decode 以缓解 OOM

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

        timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 0, ..., 999 ] -> [ 999, ... , 0 ]


        if use_skip_time and not progressive_skip:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")


        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")

        total_steps = len(timesteps) #  self.scheduler.ddim_timesteps.shape[0]
        # Notes: total_step 应该是本次调用中实际会进行的去噪步数总数, 在使用 skip 时将与 num_inference_steps 不同, 注意区分使用


        # 5. Prepare latent variable [pano]

        if total_f is None:
            total_f = frames * num_windows_f

        num_channels_latents = unet_config["params"]["in_channels"]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        total_shape = (
            batch_size,
            num_channels_latents,
            total_f,
            total_h // self.vae_scale_factor,
            total_w // self.vae_scale_factor,
        )

        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

            if use_skip_time:

                # frame_0_latent = encode_images_list_to_latent_tensor(pretrained_t2v=self.pretrained_t2v,
                #                                                      image_folder=None,
                #                                                      image_size=(total_h, total_w),
                #                                                      image_path_list=[pano_image_path])
                frame_0_latent = self.tiled_vae_encode_image(image_path=pano_image_path, image_size=(total_h, total_w))

                if progressive_skip:
                    for frame_idx, progs_skip_idx in enumerate(list(reversed(range(skip_time_step_idx)))):
                        # noised_frame_latent = self._add_noise(clear_video_latent=frame_0_latent,
                        #                                       time_step_index=total_steps-progs_skip_idx-1)
                        noised_frame_latent = self.scheduler.re_noise(x_a=frame_0_latent,
                                                                      step_a=0,
                                                                      step_b=num_inference_steps - progs_skip_idx - 1)
                        init_panorama_latent[:, :, [frame_idx]] = noised_frame_latent.clone()

                else:
                    clear_repeat_latent = torch.cat([frame_0_latent] * total_f, dim=2)
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
        # Notes: 注意 window step size (控制有overlap的滑动窗口) 与 offset step size (控制窗口起始位置不断偏移) 区别

        overlap_ratio_w = 1 - (total_w/width - 1) / (num_windows_w - 1)

        # image_window_step_size_w = int(width * (1 - overlap_ratio_w))       # TODO :CHECK
        # latent_window_step_size_w = image_window_step_size_w // self.vae_scale_factor
        latent_window_step_size_w = width / self.vae_scale_factor * (1 - overlap_ratio_w)

        image_offset_step_size_w = int((1 - overlap_ratio_w) * width / loop_step)
        latent_offset_step_size_w = image_offset_step_size_w // self.vae_scale_factor
        if num_windows_w == 1:
            image_offset_step_size_w = 0
            latent_offset_step_size_w = 0
        print(f"Shift for W: \n"
              f"overlape_ratio_w = 1 - (total_w/width - 1) / (num_windows_w - 1) = 1 - ({total_w}/{width} - 1) / ({num_windows_w} - 1) = {overlap_ratio_w}\n"
              # f"image_window_step_size_w = int(width * (1 - overlape_ratio_w)) = int({width} * (1 - {overlap_ratio_w})) = int({width * (1 - overlap_ratio_w)}) = {image_window_step_size_w}\n"
              f"image_offset_step_size_w = int(overlap_ratio_w * total_w / offset_loop_step) = int({overlap_ratio_w} * {total_w} / {loop_step}) = int({overlap_ratio_w * total_w / loop_step}) = {image_offset_step_size_w}")
        assert 0 <= overlap_ratio_w < 1, "overlap ratio for W is not legal"
        assert latent_offset_step_size_w >= 1, "latent_offset_step_size_w should > 1"

        overlap_ratio_h = 1 - (total_h/height - 1) / (num_windows_h - 1)

        # image_window_step_size_h = int(height * (1 - overlap_ratio_h))
        # latent_window_step_size_h = image_window_step_size_h // self.vae_scale_factor
        latent_window_step_size_h = height / self.vae_scale_factor * (1 - overlap_ratio_h)

        image_offset_step_size_h = int((1 - overlap_ratio_h) * height / loop_step)
        latent_offset_step_size_h = image_offset_step_size_h // self.vae_scale_factor
        if num_windows_h == 1:
            image_offset_step_size_h = 0
            latent_offset_step_size_h = 0
        print(f"Shift for H: \n"
              f"overlape_ratio_h = 1 - (total_h/height - 1) / (num_windows_h - 1) = 1 - ({total_h}/{height} - 1) / ({num_windows_h} - 1) = {overlap_ratio_h}\n"
              # f"image_window_step_size_h = int(height * (1 - overlape_ratio_h)) = int({height} * (1 - {overlap_ratio_h})) = int({width * (1 - overlap_ratio_h)}) = {image_window_step_size_h}\n"
              f"image_offset_step_size_h = int(overlap_ratio_h * total_h / offset_loop_step) = int({overlap_ratio_h} * {total_h} / {loop_step}) = int({overlap_ratio_h * total_h / loop_step}) = {image_offset_step_size_h}")
        assert 0 <= overlap_ratio_h < 1, "overlap ratio for H is not legal"
        assert latent_offset_step_size_h >= 1, "latent_offset_step_size_h should > 1"

        latent_step_size_f = frames // loop_step
        if total_f == frames:          # 不拓展时不动, 避免成环同时只有一个 window, frame 太少对画面的影响
            latent_step_size_f = 0

        assert latent_step_size_f > 0 or total_f == frames, f"[basic_sample_shift_multi_windows] loop_step {loop_step} " \
                                                             f"> frames {frames} while total_f {total_f} > frame"
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
        panorama_ring_image_tensor_handler = RingImageTensor(image_path=pano_image_path, height=total_h, width=total_w)

        bs = batch_size * num_videos_per_prompt  # ?

        _DOCK_START_INDEX = -101
        _DOCK_END_INDEX = -111

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                image_pos_left_start = ((i+begin_index_offset) % loop_step) * image_offset_step_size_w       # 第一个window的左边缘起点
                image_pos_top_start = ((i+begin_index_offset) % loop_step) * image_offset_step_size_h        # 第一个window的上边缘起点

                latent_pos_left_start = ((i+begin_index_offset) % loop_step) * latent_offset_step_size_w    # 注意区分 latent 和 image 的索引大小
                latent_pos_top_start = ((i+begin_index_offset) % loop_step) * latent_offset_step_size_h


                overlap_ratio_f = overlap_ratio_list_f[i]
                total_window_num_f = math.ceil((total_f // frames - 1) / (1 - overlap_ratio_f)) + 1

                if total_f > frames:  # 需要 shift on F

                    offset_shift_step_size_f = max(int(overlap_ratio_f * frames / loop_step_frame), 1)
                    # TODO: check, 这里为了测试强行设置了最小值为 1
                    latent_frames_begin = (i % loop_step_frame) * offset_shift_step_size_f
                    curr_shift_f_idies_list = list(range(total_window_num_f))

                    if dock_at_f:
                        curr_shift_f_idies_list = [_DOCK_START_INDEX] + curr_shift_f_idies_list + [_DOCK_END_INDEX]

                elif total_f == frames:  # 无需 shift on F
                    latent_frames_begin = 0
                    curr_shift_f_idies_list = [0]
                else:
                    print(f"total_f {total_f} should >= frames {frames} !")
                    raise ValueError

                print(f"\n"
                      f"i = {i} => +offset {i+begin_index_offset} , t = {t}")

                # reset denoised mask record
                panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))

                for shift_f_idx in curr_shift_f_idies_list:

                    for shift_w_idx in range(num_windows_w):

                        for shift_h_idx in range(num_windows_h):

                            # TODO : Docking (i2v 有需要吗?)

                            window_latent_left = latent_pos_left_start + round(shift_w_idx * latent_window_step_size_w)
                            window_latent_right = window_latent_left + latent_width
                            window_latent_top = latent_pos_top_start + round(shift_h_idx * latent_window_step_size_h)
                            window_latent_down = window_latent_top + latent_height

                            window_image_left = window_latent_left * self.vae_scale_factor  # image_pos_left_start + shift_w_idx * image_window_step_size_w
                            window_image_right = window_image_left + width
                            window_image_top = window_latent_top * self.vae_scale_factor # image_pos_top_start + shift_h_idx * image_window_step_size_h
                            window_image_down = window_image_top + height

                            window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio_f))
                            window_latent_frame_begin = window_latent_frame_begin % total_f
                            window_latent_frame_end = window_latent_frame_begin + frames

                            if dock_at_f:
                                if shift_f_idx == _DOCK_START_INDEX:
                                    if latent_frames_begin == 0:
                                        print(
                                            f"i % init_stage_loop_step = {i} % {loop_step_frame} = 0, no need for docking, skipped"
                                        )
                                        continue
                                    window_latent_frame_begin = 0
                                    window_latent_frame_end = window_latent_frame_begin + frames

                                if shift_f_idx == _DOCK_END_INDEX:
                                    if latent_frames_begin == 0:
                                        print(
                                            f"i % init_stage_loop_step = {i} % {loop_step_frame} = 0, no need for docking, skipped"
                                        )
                                        continue
                                    window_latent_frame_begin = total_f - frames
                                    window_latent_frame_end = window_latent_frame_begin + frames

                                if window_latent_frame_end > total_f:
                                    print(
                                        f"window_latent_frame_end = {window_latent_frame_end} > frame end edge = {total_f}, skipped because docking F")
                                    continue

                            window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                                                                                           pos_right=window_latent_right,
                                                                                           pos_top=window_latent_top,
                                                                                           pos_down=window_latent_down,
                                                                                           frame_begin=window_latent_frame_begin,
                                                                                           frame_end=window_latent_frame_end)

                            window_latent_prev_denoise = window_latent.clone()

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
                                # window_denoised_mask = window_denoised_mask[0, 0, [0]]
                                window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                                      latent_to_add=noised_window_latent,
                                                                      mask=window_denoised_mask,
                                                                      mix_ratio=merge_renoised_overlap_latent_ratio)
                            curr_h_factor = window_image_down / total_h
                            curr_prompt = prompt
                            if window_multi_prompt_dict is not None:
                                curr_prompt = select_prompt_from_multi_prompt_dict_by_factor(prompt_dict=window_multi_prompt_dict,
                                                                                             factor=curr_h_factor)
                                text_emb = self.pretrained_t2v.get_learned_conditioning([curr_prompt])

                            img_emb = panorama_ring_image_tensor_handler.get_encoded_image_cond(pretrained_t2v=self.pretrained_t2v,
                                                                                                pos_left=window_image_left,
                                                                                                pos_right=window_image_right,
                                                                                                pos_top=window_image_top,
                                                                                                pos_down=window_image_down)
                            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
                            cond = {"c_crossattn": [imtext_cond], "fps": fps}

                            print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
                                  f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
                                  f"window image: h[{window_image_top} - {window_image_down}] w[{window_image_left} - {window_image_right}]"
                                  f"curr prompt ({curr_h_factor}): {curr_prompt}")

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

                            if merge_prev_denoised_ratio_list is not None and i < total_steps - 1:     # Notes: 加权平均 overlap 部分的去噪结果

                                merge_prev_denoised_ratio = merge_prev_denoised_ratio_list[i]
                                window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                                      latent_to_add=window_latent_prev_denoise,
                                                                      mask=window_denoised_mask,
                                                                      mix_ratio=merge_prev_denoised_ratio)

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
            denoised_chunk_num = 16
            denoised_chunked = list(torch.chunk(denoised, denoised_chunk_num, dim=4))
            denoised_chunked_cat_list = [denoised_chunked[-1]] + denoised_chunked + [denoised_chunked[0]]
            denoised = torch.cat(denoised_chunked_cat_list, dim=4)

            # videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)      # doubt: VAE 能正常 decode 超宽的 latent 吗 ?

            if tiled_decode and not (total_w // width <= 2 and total_h // height <= 2):
                h_tile_num = 2
                w_tile_num = 1
                overlap_h = min(4 * h_tile_num, 32)
                overlap_w = min(4 * w_tile_num, 32)
                videos = self.tiled_vae_decode_linear_merge(denoised=denoised,
                                                            h_tile_num=h_tile_num, w_tile_num=w_tile_num,
                                                            overlap_h=overlap_h, overlap_w=overlap_w)
            else:
                video_frames_list = []

                for frame_idx in range(total_f):
                    denoised_frame_latent = denoised[:, :, [frame_idx]]
                    video_frames_tensor = self.pretrained_t2v.decode_first_stage_2DAE(denoised_frame_latent)
                    video_frames_list.append(video_frames_tensor)

                videos = torch.cat(video_frames_list, dim=2)

            videos_chunked = torch.chunk(videos, denoised_chunk_num+2, dim=4)
            videos = torch.cat(videos_chunked[1:-1], dim=4)

        else:
            videos = denoised

        return videos, denoised

    @torch.no_grad()
    def tiled_vae_decode(self, denoised: torch.Tensor, h_tile_num=4, w_tile_num=4, overlap_h=32, overlap_w=32):
        """
        基于2D分块进行VAE decode的函数，用于解决高分辨率latent在解码阶段OOM问题。
        通过在每个分块(tile)加入overlap的方式，避免在拼接时出现明显缝隙。

        参数：
            denoised: [B, C, F, H, W]的latent张量
            h_tile_num: 高度方向的瓦片数
            w_tile_num: 宽度方向的瓦片数
            overlap_h: 瓦片在高度方向的重叠大小(latent空间下)
            overlap_w: 瓦片在宽度方向的重叠大小(latent空间下)

        返回：
            videos: [B, 3, F, H_decoded, W_decoded]的图像张量
                    其中H_decoded = H * self.vae_scale_factor, W_decoded = W * self.vae_scale_factor
        """

        B, C, F, H, W = denoised.shape
        # VAE放大倍数(假设已知)
        scale_factor = self.vae_scale_factor
        H_dec = H * scale_factor
        W_dec = W * scale_factor

        # 根据分块数量确定每个瓦片的宽高（不考虑整除问题，这里假设可以整除）
        tile_h = H // h_tile_num
        tile_w = W // w_tile_num

        # 最终结果初始化
        # 我们将最终decode出来的图像按照瓦片进行合成
        # 先用一个空tensor存放最终结果
        videos = torch.zeros(B, 3, F, H_dec, W_dec, device=denoised.device, dtype=denoised.dtype)

        # 为了简化，我们逐瓦片进行decode
        # 对于每个瓦片[i, j]:
        # 瓦片原始范围：
        #   高度方向: h_start = i*tile_h, h_end = (i+1)*tile_h
        #   宽度方向: w_start = j*tile_w, w_end = (j+1)*tile_w
        #
        # 为了有overlap，我们向四周扩展
        # 但是边界瓦片可能无法完整扩展，因此需要clip到实际范围
        #
        # overlap区域在拼接时会被截断掉, 最终只保留中间原定瓦片区域对应的decode结果

        for i in range(h_tile_num):
            for j in range(w_tile_num):
                h_start = i * tile_h
                h_end = (i + 1) * tile_h
                w_start = j * tile_w
                w_end = (j + 1) * tile_w

                # 带overlap的范围
                h_start_ov = max(h_start - overlap_h, 0)
                h_end_ov = min(h_end + overlap_h, H)
                w_start_ov = max(w_start - overlap_w, 0)
                w_end_ov = min(w_end + overlap_w, W)

                denoised_tile = denoised[:, :, :, h_start_ov:h_end_ov, w_start_ov:w_end_ov]

                # 对该tile进行VAE decode
                # decode_first_stage_2DAE函数输入为[B, C, F, h_sub, w_sub]
                # 输出为[B, 3, F, h_sub_dec, w_sub_dec]
                tile_decoded = self.pretrained_t2v.decode_first_stage_2DAE(denoised_tile)

                # tile_decoded形状为[B, 3, F, (h_sub * scale_factor), (w_sub * scale_factor)]
                # 我们只取出对应于原tile范围扩张前的位置，即在decode后的空间中也同样截取掉overlap对应的区域
                # 对应截取时，需要将overlap区域乘以scale_factor
                top_cut = (h_start - h_start_ov) * scale_factor
                left_cut = (w_start - w_start_ov) * scale_factor
                bottom_cut = tile_decoded.shape[3] - ((h_end_ov - h_end) * scale_factor)
                right_cut = tile_decoded.shape[4] - ((w_end_ov - w_end) * scale_factor)

                tile_decoded_cropped = tile_decoded[:, :, :, top_cut:bottom_cut, left_cut:right_cut]

                # 将截取后的结果放回videos的对应位置
                videos[:, :, :, h_start * scale_factor:h_end * scale_factor,
                w_start * scale_factor:w_end * scale_factor] = tile_decoded_cropped

        return videos

    @torch.no_grad()
    def tiled_vae_decode_linear_merge(self, denoised: torch.Tensor, h_tile_num=4, w_tile_num=4, overlap_h=8, overlap_w=8):
        """
        基于2D分块进行VAE decode的函数，使用融合机制避免拼接缝隙。
        通过在每个分块(tile)加入overlap的方式，并在拼接时进行加权融合，避免拼接处的明显缝隙。

        参数：
            denoised: [B, C, F, H, W]的latent张量
            h_tile_num: 高度方向的瓦片数
            w_tile_num: 宽度方向的瓦片数
            overlap_h: 瓦片在高度方向的重叠大小(latent空间下)
            overlap_w: 瓦片在宽度方向的重叠大小(latent空间下)

        返回：
            videos: [B, 3, F, H_decoded, W_decoded]的图像张量
                    其中H_decoded = H * self.vae_scale_factor, W_decoded = W * self.vae_scale_factor
        """

        def create_blend_mask(tile_height, tile_width, overlap_h, overlap_w,
                              is_top_edge, is_bottom_edge, is_left_edge, is_right_edge,
                              device):
            """
            创建融合mask，只对非最外层边界做渐变。
            对外层边界保持为1，以免出现黑边。
            """

            mask = torch.ones((tile_height, tile_width), device=device, dtype=torch.float32)

            # 如果不是顶边界，则对top overlap区做渐变，从0到1
            if overlap_h > 0 and not is_top_edge:
                top_weights = torch.linspace(0.5, 1, overlap_h, device=device)
                mask[:overlap_h, :] = mask[:overlap_h, :] * top_weights.unsqueeze(1)

            # 如果不是底边界，则对bottom overlap区做渐变，从1到0
            if overlap_h > 0 and not is_bottom_edge:
                bottom_weights = torch.linspace(1, 0.5, overlap_h, device=device)
                mask[-overlap_h:, :] = mask[-overlap_h:, :] * bottom_weights.unsqueeze(1)

            # 如果不是左边界，则对left overlap区做渐变，从0到1
            if overlap_w > 0 and not is_left_edge:
                left_weights = torch.linspace(0.5, 1, overlap_w, device=device)
                mask[:, :overlap_w] = mask[:, :overlap_w] * left_weights.unsqueeze(0)

            # 如果不是右边界，则对right overlap区做渐变，从1到0
            if overlap_w > 0 and not is_right_edge:
                right_weights = torch.linspace(1, 0.5, overlap_w, device=device)
                mask[:, -overlap_w:] = mask[:, -overlap_w:] * right_weights.unsqueeze(0)

            return mask

        B, C, F, H, W = denoised.shape
        scale_factor = self.vae_scale_factor
        H_dec = H * scale_factor
        W_dec = W * scale_factor

        tile_h = H // h_tile_num
        tile_w = W // w_tile_num

        device = denoised.device
        videos_accumulator = torch.zeros(B, 3, F, H_dec, W_dec, device=device, dtype=torch.float32)
        weight_accumulator = torch.zeros(B, 1, F, H_dec, W_dec, device=device, dtype=torch.float32)

        for i in range(h_tile_num):
            for j in range(w_tile_num):
                h_start = i * tile_h
                h_end = (i + 1) * tile_h
                w_start = j * tile_w
                w_end = (j + 1) * tile_w

                # 带overlap的范围（latent）
                h_start_ov = max(h_start - overlap_h, 0)
                h_end_ov = min(h_end + overlap_h, H)
                w_start_ov = max(w_start - overlap_w, 0)
                w_end_ov = min(w_end + overlap_w, W)

                denoised_tile = denoised[:, :, :, h_start_ov:h_end_ov, w_start_ov:w_end_ov]
                tile_decoded = self.pretrained_t2v.decode_first_stage_2DAE(denoised_tile)

                # decode后大小
                h_sub = h_end_ov - h_start_ov
                w_sub = w_end_ov - w_start_ov
                h_sub_dec = h_sub * scale_factor
                w_sub_dec = w_sub * scale_factor

                # 截取掉overlap以获得原定tile大小的decoded结果
                top_cut = (h_start - h_start_ov) * scale_factor
                left_cut = (w_start - w_start_ov) * scale_factor
                bottom_cut = h_sub_dec - ((h_end_ov - h_end) * scale_factor)
                right_cut = w_sub_dec - ((w_end_ov - w_end) * scale_factor)

                tile_decoded_cropped = tile_decoded[:, :, :, top_cut:bottom_cut, left_cut:right_cut]

                final_h_start = h_start * scale_factor
                final_h_end = h_end * scale_factor
                final_w_start = w_start * scale_factor
                final_w_end = w_end * scale_factor

                # 根据瓦片位置，判断哪些边是外层边界
                is_top_edge = (i == 0)
                is_bottom_edge = (i == h_tile_num - 1)
                is_left_edge = (j == 0)
                is_right_edge = (j == w_tile_num - 1)

                overlap_h_dec = overlap_h * scale_factor
                overlap_w_dec = overlap_w * scale_factor
                tile_height_dec = tile_decoded_cropped.shape[3]
                tile_width_dec = tile_decoded_cropped.shape[4]

                mask_2d = create_blend_mask(
                    tile_height_dec, tile_width_dec,
                    overlap_h_dec, overlap_w_dec,
                    is_top_edge, is_bottom_edge,
                    is_left_edge, is_right_edge,
                    device=device
                )

                mask = mask_2d.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,h,w]
                mask = mask.expand(B, 1, F, tile_height_dec, tile_width_dec)

                videos_accumulator[:, :, :, final_h_start:final_h_end,
                final_w_start:final_w_end] += tile_decoded_cropped * mask
                weight_accumulator[:, :, :, final_h_start:final_h_end, final_w_start:final_w_end] += mask

        # 避免除零，对无重叠区域权重为0的情况应不存在，但以防万一先做处理
        # 对于任何权重为0的区域，说明该区域没有瓦片覆盖，可以设为0或跳过。
        # 一般正常分块不会有这种情况。
        # weight_accumulator[weight_accumulator == 0] = 1.0

        videos = videos_accumulator / weight_accumulator
        return videos


    @torch.no_grad()
    def tiled_vae_encode_image(self, image_path, image_size):
        image_tensor = _load_and_preprocess_image(image_path, image_size)
        image_tensor = image_tensor.unsqueeze(1).unsqueeze(0).to(dtype=self.pretrained_t2v.dtype, device=self.pretrained_t2v.device)
        encoded_latent = self.tiled_vae_encode_tensor(image_tensor=image_tensor)
        return encoded_latent

    def tiled_vae_encode_tensor(self, image_tensor: torch.Tensor, h_tile_num=4, w_tile_num=4, overlap_h=32, overlap_w=32):
        """
        基于2D分块进行VAE encode的函数，用于解决高分辨率图像在编码阶段OOM问题。
        通过在每个分块(tile)加入overlap的方式，避免在拼接时出现明显缝隙。

        参数：
            image_tensor: [B, 3, F, H_decoded, W_decoded] 的图像张量
            h_tile_num: 高度方向的瓦片数
            w_tile_num: 宽度方向的瓦片数
            overlap_h: 瓦片在高度方向的重叠大小（latent空间下）
            overlap_w: 瓦片在宽度方向的重叠大小（latent空间下）

        返回：
            img_latent: [B, C, F, H, W] 的latent张量
        """

        B, C_img, F, H_dec, W_dec = image_tensor.shape
        scale_factor = self.vae_scale_factor
        H_latent = H_dec // scale_factor
        W_latent = W_dec // scale_factor

        # 计算每个瓦片在图像空间的尺寸
        tile_h_latent = H_latent // h_tile_num
        tile_w_latent = W_latent // w_tile_num
        tile_h_image = tile_h_latent * scale_factor
        tile_w_image = tile_w_latent * scale_factor

        # 计算图像空间的重叠大小
        overlap_h_image = overlap_h * scale_factor
        overlap_w_image = overlap_w * scale_factor

        # 初始化latent张量和计数张量用于平均
        device = image_tensor.device
        dtype = image_tensor.dtype
        img_latent = torch.zeros((B, 4, F, H_latent, W_latent), device=device,
                                 dtype=torch.float32)
        count = torch.zeros((B, 1, 1, H_latent, W_latent), device=device, dtype=torch.float32)

        for i in range(h_tile_num):
            for j in range(w_tile_num):
                # 计算当前瓦片在图像空间的起始和结束位置
                h_start_image = i * tile_h_image
                h_end_image = (i + 1) * tile_h_image
                w_start_image = j * tile_w_image
                w_end_image = (j + 1) * tile_w_image

                # 添加重叠区域，确保不超出图像边界
                h_start_ov_image = max(h_start_image - overlap_h_image, 0)
                h_end_ov_image = min(h_end_image + overlap_h_image, H_dec)
                w_start_ov_image = max(w_start_image - overlap_w_image, 0)
                w_end_ov_image = min(w_end_image + overlap_w_image, W_dec)

                # 提取带有重叠区域的图像瓦片
                image_tile = image_tensor[:, :, :, h_start_ov_image:h_end_ov_image, w_start_ov_image:w_end_ov_image]

                # 对该瓦片进行VAE编码
                latent_tile = self.pretrained_t2v.encode_first_stage_2DAE(image_tile)  # 假设输出形状为 [B, C, F, h_sub, w_sub]

                # 计算在latent空间中需要裁剪的区域
                top_cut = (h_start_image - h_start_ov_image) // scale_factor
                left_cut = (w_start_image - w_start_ov_image) // scale_factor
                bottom_cut = latent_tile.shape[3] - ((h_end_ov_image - h_end_image) // scale_factor)
                right_cut = latent_tile.shape[4] - ((w_end_ov_image - w_end_image) // scale_factor)

                latent_tile_cropped = latent_tile[:, :, :, top_cut:bottom_cut, left_cut:right_cut]

                # 计算latent空间中拼接的位置
                h_start_latent = i * tile_h_latent
                h_end_latent = (i + 1) * tile_h_latent
                w_start_latent = j * tile_w_latent
                w_end_latent = (j + 1) * tile_w_latent

                # 将裁剪后的latent瓦片累加到最终的latent张量中
                img_latent[:, :, :, h_start_latent:h_end_latent, w_start_latent:w_end_latent] += latent_tile_cropped

                # 更新计数张量，用于后续平均
                count[:, :, :, h_start_latent:h_end_latent, w_start_latent:w_end_latent] += 1

        # 防止除以零
        count = torch.clamp(count, min=1.0)

        # 对重叠区域进行平均
        img_latent = img_latent / count

        return img_latent

    # @torch.no_grad()
    # def basic_sample_shift_multi_windows_upscale(
    #         self,
    #         prompt: Union[str, List[str]] = None,
    #         img_cond_path: Union[str, List[str]] = None,
    #         height: Optional[int] = 320,
    #         width: Optional[int] = 512,
    #         frames: int = 16,
    #         fps: int = 16,
    #         guidance_scale: float = 7.5,
    #         num_videos_per_prompt: Optional[int] = 1,
    #         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # TODO: 已给出部分的 noise 也应该保持与生产时使用的相同 ?
    #
    #         init_panorama_latent: torch.Tensor = None,  # 包含整个 panorama 的 latent
    #         total_w: int = None,
    #         total_h: int = None,                        # 允许任意自定义宽高, window 间可以重叠
    #         total_f: int = None,
    #         num_windows_w: int = None,                  # W 方向 window 数
    #         num_windows_h: int = None,
    #         num_windows_f: int = None,                  # 总帧数是 frames (16)  * num_windows_f
    #         loop_step: int = None,                      # 应大于1, 越小 window 滑动越快      # doubt: 是否应该也分 h / w
    #
    #         begin_index_offset: int = 0,                # 计算每个 timestep 对应的H,W,F起始点前加入 offset,
    #                                                     # 可能缓解边缘接缝 gap 问题
    #
    #         pano_image_path: str = None,
    #
    #         latents: Optional[torch.FloatTensor] = None,
    #         num_inference_steps: int = 4,
    #         prompt_embeds: Optional[torch.FloatTensor] = None,
    #         output_type: Optional[str] = "pil",
    #
    #         merge_renoised_overlap_latent_ratio:float = 1,
    #
    #         use_pre_denoise: bool = False,
    #         clear_pre_denoised_latent: torch.Tensor = None,
    #         merge_predenoise_ratio_list: list[float] = None,
    #
    #         use_skip_time=False,
    #         skip_time_step_idx=None,
    #         progressive_skip=False,
    #         **kwargs
    # ):
    #     unet_config = self.model_config["params"]["unet_config"]
    #     # 0. Default height and width to unet
    #     frames = self.pretrained_t2v.temporal_length if frames < 0 else frames
    #
    #     # 2. Define call parameters
    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #         prompt = [prompt]
    #         if isinstance(img_cond_path, str):
    #             img_cond_path = [img_cond_path]
    #         else:
    #             assert len(img_cond_path) == 1, "[basic_sample] cond img should have same amount as text prompts"
    #     elif prompt is not None and isinstance(prompt, list):
    #         assert len(prompt) == len(img_cond_path), "[basic_sample] cond img should have same amount as text prompts"
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]
    #
    #     device = self._execution_device
    #
    #     # 3. Encode input prompt
    #     cond_images = self._load_imgs_from_paths(img_path_list=img_cond_path, height=height, width=width)
    #     cond_images = cond_images.to(self.pretrained_t2v.device)
    #     img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
    #     text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)
    #
    #     imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    #     cond = {"c_crossattn": [imtext_cond], "fps": fps}
    #
    #
    #     # 3.5 Prepare CFG if used
    #     if guidance_scale != 1.0:
    #         uncond_type = self.pretrained_t2v.uncond_type
    #         if uncond_type == "empty_seq":
    #             prompts = batch_size * [""]
    #             # prompts = N * T * [""]  ## if is_imgbatch=True
    #             uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
    #         elif uncond_type == "zero_embed":
    #             c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
    #             uc_emb = torch.zeros_like(c_emb)
    #         else:
    #             raise NotImplementedError()
    #
    #         ## process image embedding token
    #         if hasattr(self.pretrained_t2v, 'embedder'):
    #             uc_img = torch.zeros(batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor).to(
    #                 self.pretrained_t2v.device)
    #             ## img: b c h w >> b l c
    #             uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
    #             uc_emb = torch.cat([uc_emb, uc_img], dim=1)
    #
    #         if isinstance(cond, dict):
    #             uncond = {key: cond[key] for key in cond.keys()}
    #             uncond.update({'c_crossattn': [uc_emb]})
    #         else:
    #             uncond = uc_emb
    #     else:
    #         uncond = None
    #
    #     # 4. Prepare timesteps
    #     self.scheduler.make_schedule(num_inference_steps)  # set_timesteps(num_inference_steps)   # , lcm_origin_steps)
    #
    #     timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 0, ..., 999 ] -> [ 999, ... , 0 ]
    #
    #
    #     if use_skip_time and not progressive_skip:
    #         timesteps = timesteps[skip_time_step_idx:]
    #         print(f"skip : {skip_time_step_idx}")
    #
    #
    #     print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
    #     print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")
    #
    #     total_steps = len(timesteps) #  self.scheduler.ddim_timesteps.shape[0]
    #     # Notes: total_step 应该是本次调用中实际会进行的去噪步数总数, 在使用 skip 时将与 num_inference_steps 不同, 注意区分使用
    #
    #     # 5. Prepare latent variable [pano]
    #     num_channels_latents = unet_config["params"]["in_channels"]
    #     latent_height = height // self.vae_scale_factor
    #     latent_width = width // self.vae_scale_factor
    #     total_shape = (
    #         batch_size,
    #         num_channels_latents,
    #         frames * num_windows_f,
    #         total_h // self.vae_scale_factor,
    #         total_w // self.vae_scale_factor,
    #     )
    #
    #     if init_panorama_latent is None:
    #
    #         init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)
    #
    #         if use_skip_time:
    #
    #             if use_pre_denoise:
    #
    #                 _basic_latent_shape = (
    #                     batch_size,
    #                     num_channels_latents,
    #                     frames,
    #                     latent_height,
    #                     latent_width,
    #                 )
    #
    #                 assert list(clear_pre_denoised_latent.shape) == list(_basic_latent_shape), \
    #                     f"[basic_sample_shift_multi_windows] " \
    #                     f"clear_pre_denoised_latent shape :{clear_pre_denoised_latent}" \
    #                     f"not equal to _basic_latent_shape: {_basic_latent_shape}"
    #
    #                 resized_latent = resize_video_latent(input_latent=clear_pre_denoised_latent.clone(), mode="bicubic",
    #                                                      target_height=latent_height * num_windows_h,
    #                                                      target_width=latent_width * num_windows_w)
    #
    #             else:
    #                 warnings.warn("Using basic_sample_shift_multi_windows_upscale() but use_skip_time is False !!")
    #             # else:
    #             #
    #             # frame_0_latent = encode_images_list_to_latent_tensor(pretrained_t2v=self.pretrained_t2v,
    #             #                                                      image_folder=None,
    #             #                                                      image_size=(total_h, total_w),
    #             #                                                      image_path_list=[pano_image_path])
    #             # if progressive_skip:
    #             #     for frame_idx, progs_skip_idx in enumerate(list(reversed(range(skip_time_step_idx)))):
    #             #         # noised_frame_latent = self._add_noise(clear_video_latent=frame_0_latent,
    #             #         #                                       time_step_index=total_steps-progs_skip_idx-1)
    #             #         noised_frame_latent = self.scheduler.re_noise(x_a=frame_0_latent,
    #             #                                                       step_a=0,
    #             #                                                       step_b=num_inference_steps - progs_skip_idx - 1)
    #             #         init_panorama_latent[:, :, [frame_idx]] = noised_frame_latent.clone()
    #             #
    #             # else:
    #             #     clear_repeat_latent = torch.cat([frame_0_latent] * frames * num_windows_f, dim=2)
    #             #     # init_panorama_latent = self._add_noise(clear_video_latent=clear_repeat_latent,
    #             #     #                                        time_step_index=total_steps-1)
    #             #     init_panorama_latent = self.scheduler.re_noise(x_a=clear_repeat_latent,
    #             #                                                    step_a=0,
    #             #                                                    step_b=total_steps-1)
    #
    #     else:
    #         print("[basic_sample_shift_multi_windows] using given init latent")
    #         assert init_panorama_latent.shape == total_shape, f"[basic_sample_shift_multi_windows] " \
    #                                                           f"init_panorama_latent shape {init_panorama_latent.shape}" \
    #                                                           f"does not match" \
    #                                                           f"desired shape {total_shape}"
    #         init_panorama_latent = init_panorama_latent.clone()
    #
    #
    #     panorama_ring_latent_handler = RingLatent(init_latent=init_panorama_latent)
    #     panorama_ring_latent_denoised_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))
    #
    #     # define window shift
    #     # Notes: 注意 window step size (控制有overlap的滑动窗口) 与 offset step size (控制窗口起始位置不断偏移) 区别
    #
    #     overlap_ratio_w = 1 - (total_w/width - 1) / (num_windows_w - 1)
    #
    #     image_window_step_size_w = int(width * (1 - overlap_ratio_w))       # TODO :CHECK
    #     latent_window_step_size_w = image_window_step_size_w // self.vae_scale_factor
    #
    #     image_offset_step_size_w = int((1 - overlap_ratio_w) * total_w / loop_step)
    #     latent_offset_step_size_w = image_offset_step_size_w // self.vae_scale_factor
    #     if num_windows_w == 1:
    #         image_offset_step_size_w = 0
    #         latent_offset_step_size_w = 0
    #     print(f"Shift for W: \n"
    #           f"overlape_ratio_w = 1 - (total_w/width - 1) / (num_windows_w - 1) = 1 - ({total_w}/{width} - 1) / ({num_windows_w} - 1) = {overlap_ratio_w}\n"
    #           f"image_window_step_size_w = int(width * (1 - overlape_ratio_w)) = int({width} * (1 - {overlap_ratio_w})) = int({width * (1 - overlap_ratio_w)}) = {image_window_step_size_w}\n"
    #           f"image_offset_step_size_w = int(overlap_ratio_w * total_w / offset_loop_step) = int({overlap_ratio_w} * {total_w} / {loop_step}) = int({overlap_ratio_w * total_w / loop_step}) = {image_offset_step_size_w}")
    #     assert 0 <= overlap_ratio_w < 1, "overlap ratio for W is not legal"
    #
    #
    #     overlap_ratio_h = 1 - (total_h/height - 1) / (num_windows_h - 1)
    #
    #     image_window_step_size_h = int(height * (1 - overlap_ratio_h))
    #     latent_window_step_size_h = image_window_step_size_h // self.vae_scale_factor
    #
    #     image_offset_step_size_h = int((1 - overlap_ratio_w) * total_h / loop_step)
    #     latent_offset_step_size_h = image_offset_step_size_h // self.vae_scale_factor
    #     if num_windows_h == 1:
    #         image_offset_step_size_h = 0
    #         latent_offset_step_size_h = 0
    #     print(f"Shift for H: \n"
    #           f"overlape_ratio_h = 1 - (total_h/height - 1) / (num_windows_h - 1) = 1 - ({total_h}/{height} - 1) / ({num_windows_h} - 1) = {overlap_ratio_h}\n"
    #           f"image_window_step_size_h = int(height * (1 - overlape_ratio_h)) = int({height} * (1 - {overlap_ratio_h})) = int({width * (1 - overlap_ratio_h)}) = {image_window_step_size_h}\n"
    #           f"image_offset_step_size_h = int(overlap_ratio_h * total_h / offset_loop_step) = int({overlap_ratio_h} * {total_h} / {loop_step}) = int({overlap_ratio_h * total_h / loop_step}) = {image_offset_step_size_h}")
    #     assert 0 <= overlap_ratio_h < 1, "overlap ratio for H is not legal"
    #
    #     latent_step_size_f = frames // loop_step
    #     if num_windows_f == 1:          # 不拓展时不动, 避免成环同时只有一个 window, frame 太少对画面的影响
    #         latent_step_size_f = 0
    #
    #     assert latent_step_size_f > 0 or num_windows_f == 1, f"[basic_sample_shift_multi_windows] loop_step {loop_step} " \
    #                                                          f"> frames {frames} while num_windows_f {num_windows_f} > 0"
    #     # TODO: 增加精确适配 step_size 为 小数时的 SW 位置分配
    #     #  loop_step = 100
    #     #  latent_step_size_f = 16 / loop_step
    #     #  prev = 1
    #     #  for i in range(48):
    #     #      xx = (i % loop_step) * latent_step_size_f
    #     #      if int(xx) == prev:
    #     #          pp = int(xx) + (int(xx)-prev) - 1
    #     #      else:
    #     #          pp = int(xx)
    #     #      prev = pp
    #     #      print(f"[{i}({i % loop_step})] {round(xx, 4)}: \t int -> {int(xx)}, round -> {round(xx)}, pp={pp}")
    #
    #     # prepare window image cond
    #     panorama_ring_image_tensor_handler = RingImageTensor(image_path=pano_image_path, height=total_h, width=total_w)
    #
    #     bs = batch_size * num_videos_per_prompt  # ?
    #
    #     # 6. DDIM Sampling Loop
    #     with self.progress_bar(total=len(timesteps)) as progress_bar:
    #         for i, t in enumerate(timesteps):
    #
    #             image_pos_left_start = ((i+begin_index_offset) % loop_step) * image_offset_step_size_w       # 第一个window的左边缘起点
    #             image_pos_top_start = ((i+begin_index_offset) % loop_step) * image_offset_step_size_h        # 第一个window的上边缘起点
    #
    #             latent_pos_left_start = ((i+begin_index_offset) % loop_step) * latent_offset_step_size_w    # 注意区分 latent 和 image 的索引大小
    #             latent_pos_top_start = ((i+begin_index_offset) % loop_step) * latent_offset_step_size_h
    #
    #             latent_frames_begin = ((i+begin_index_offset) % loop_step) * latent_step_size_f
    #
    #             print(f"\n"
    #                   f"i = {i} => +offset {i+begin_index_offset} , t = {t}")
    #
    #             # reset denoised mask record
    #             panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))
    #
    #
    #
    #             if merge_predenoise_ratio_list is not None and resized_latent is not None:
    #
    #                 # resized_latent = resize_video_latent(input_latent=clear_pre_denoised_latent.clone(), mode="bilinear",
    #                 #                                      target_height=latent_height * num_windows_h,
    #                 #                                      target_width=latent_width * num_windows_w).clone()
    #
    #                 assert len(merge_predenoise_ratio_list) == len(timesteps), f"merge_predenoise_ratio_list " \
    #                                                                            f"({len(merge_predenoise_ratio_list)}) " \
    #                                                                            f"should have same length as timesteps" \
    #                                                                            f"({len(timesteps)})"
    #
    #
    #                 curr_merge_ratio = merge_predenoise_ratio_list[i]
    #                 print(f"merging residual latent: {round(curr_merge_ratio, 3)} * curr + {round(1.0-curr_merge_ratio, 3)} * noised_resized")
    #
    #                 curr_latent = panorama_ring_latent_handler.torch_latent
    #                 # noised_resized_latent = self._add_noise(clear_video_latent=resized_latent, time_step_index=total_steps-i-1)
    #                 noised_resized_latent = self.scheduler.re_noise(x_a=resized_latent.clone(),
    #                                                                 step_a=0,
    #                                                                 step_b=total_steps - i - 1)
    #                 # noised_resized_latent = self.scheduler.re_noise_x0(x_start=resized_latent, timestep=total_steps - i - 1)
    #
    #                 mixed_residual_latent = curr_merge_ratio * curr_latent + (1.0-curr_merge_ratio) * noised_resized_latent
    #                 panorama_ring_latent_handler.torch_latent = mixed_residual_latent.clone()
    #
    #
    #
    #             for shift_f_idx in range(num_windows_f):
    #
    #                 for shift_w_idx in range(num_windows_w):
    #
    #                     for shift_h_idx in range(num_windows_h):
    #
    #                         # TODO : Docking (i2v 有需要吗?)
    #
    #                         window_image_left = image_pos_left_start + shift_w_idx * image_window_step_size_w
    #                         window_image_right = window_image_left + width
    #                         window_image_top = image_pos_top_start + shift_h_idx * image_window_step_size_h
    #                         window_image_down = window_image_top + height
    #
    #                         window_latent_left = latent_pos_left_start + shift_w_idx * latent_window_step_size_w
    #                         window_latent_right = window_latent_left + latent_width
    #                         window_latent_top = latent_pos_top_start + shift_h_idx * latent_window_step_size_h
    #                         window_latent_down = window_latent_top + latent_height
    #
    #                         window_latent_frame_begin = latent_frames_begin + shift_f_idx * frames
    #                         window_latent_frame_end = window_latent_frame_begin + frames
    #
    #                         window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
    #                                                                                        pos_right=window_latent_right,
    #                                                                                        pos_top=window_latent_top,
    #                                                                                        pos_down=window_latent_down,
    #                                                                                        frame_begin=window_latent_frame_begin,
    #                                                                                        frame_end=window_latent_frame_end)
    #                         window_denoised_mask = panorama_ring_mask_handler.get_window_latent(pos_left=window_latent_left,
    #                                                                                             pos_right=window_latent_right,
    #                                                                                             pos_top=window_latent_top,
    #                                                                                             pos_down=window_latent_down,
    #                                                                                             frame_begin=window_latent_frame_begin,
    #                                                                                             frame_end=window_latent_frame_end)
    #
    #                         if merge_renoised_overlap_latent_ratio is not None and i < total_steps - 1:
    #
    #                             noised_window_latent = self.scheduler.re_noise(x_a=window_latent.clone(),
    #                                                                            step_a=total_steps - i - 1 - 1,
    #                                                                            step_b=total_steps - i - 1)
    #                             window_denoised_mask = window_denoised_mask[0, 0, [0]]
    #                             window_latent = mix_latents_with_mask(latent_1=window_latent,
    #                                                                   latent_to_add=noised_window_latent,
    #                                                                   mask=window_denoised_mask,
    #                                                                   mix_ratio=merge_renoised_overlap_latent_ratio)
    #
    #
    #                         img_emb = panorama_ring_image_tensor_handler.get_encoded_image_cond(pretrained_t2v=self.pretrained_t2v,
    #                                                                                             pos_left=window_image_left,
    #                                                                                             pos_right=window_image_right,
    #                                                                                             pos_top=window_image_top,
    #                                                                                             pos_down=window_image_down)
    #                         imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    #                         cond = {"c_crossattn": [imtext_cond], "fps": fps}
    #
    #                         print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
    #                               f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
    #                               f"window image: h[{window_image_top} - {window_image_down}] w[{window_image_left} - {window_image_right}]")
    #
    #                         kwargs.update({"clean_cond": True})
    #
    #                         ts = torch.full((bs,), t, device=device, dtype=torch.long)  # [1]
    #
    #                         model_pred_cond = self.pretrained_t2v.model(  # self.unet(
    #                             window_latent,
    #                             ts,
    #                             **cond,
    #                             # timestep_cond=w_embedding.to(self.dtype),
    #                             curr_time_steps=ts,
    #                             temporal_length=frames,
    #                             **kwargs  # doubt: **kwargs?
    #                         )
    #
    #                         if guidance_scale != 1.0:
    #
    #                             model_pred_uncond = self.pretrained_t2v.model(  # self.unet(
    #                                 window_latent,
    #                                 ts,
    #                                 **uncond,
    #                                 # timestep_cond=w_embedding.to(self.dtype),
    #                                 curr_time_steps=ts,
    #                                 temporal_length=frames,
    #                                 **kwargs
    #                             )
    #
    #                             model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
    #
    #                         else:
    #                             model_pred = model_pred_cond
    #
    #                         index = total_steps - i - 1  # Notes: 之前的 timesteps 进行了 flip, 所以要再倒转
    #
    #                         window_latent, denoised = self.scheduler.ddim_step(sample=window_latent, noise_pred=model_pred,
    #                                                                            indices=[index] * window_latent.shape[2])
    #
    #                         panorama_ring_latent_handler.set_window_latent(window_latent,
    #                                                                        pos_left=window_latent_left,
    #                                                                        pos_right=window_latent_right,
    #                                                                        pos_top=window_latent_top,
    #                                                                        pos_down=window_latent_down,
    #                                                                        frame_begin=window_latent_frame_begin,
    #                                                                        frame_end=window_latent_frame_end)
    #
    #                         panorama_ring_latent_denoised_handler.set_window_latent(denoised,
    #                                                                                 pos_left=window_latent_left,
    #                                                                                 pos_right=window_latent_right,
    #                                                                                 pos_top=window_latent_top,
    #                                                                                 pos_down=window_latent_down,
    #                                                                                 frame_begin=window_latent_frame_begin,
    #                                                                                 frame_end=window_latent_frame_end)
    #
    #                         new_window_denoised_mask = torch.ones_like(window_latent, dtype=window_latent.dtype, device=window_latent.device)
    #                         panorama_ring_mask_handler.set_window_latent(new_window_denoised_mask,
    #                                                                      pos_left=window_latent_left,
    #                                                                      pos_right=window_latent_right,
    #                                                                      pos_top=window_latent_top,
    #                                                                      pos_down=window_latent_down,
    #                                                                      frame_begin=window_latent_frame_begin,
    #                                                                      frame_end=window_latent_frame_end)
    #
    #             progress_bar.update()
    #
    #     denoised = panorama_ring_latent_denoised_handler.torch_latent.clone().to(device=init_panorama_latent.device)
    #
    #     if not output_type == "latent":
    #         denoised_chunk_num = 16
    #         denoised_chunked = list(torch.chunk(denoised, denoised_chunk_num, dim=4))
    #         denoised_chunked_cat_list = [denoised_chunked[-1]] + denoised_chunked + [denoised_chunked[0]]
    #         denoised = torch.cat(denoised_chunked_cat_list, dim=4)
    #
    #         # videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)      # doubt: VAE 能正常 decode 超宽的 latent 吗 ?
    #
    #         video_frames_list = []
    #
    #         for frame_idx in range(frames * num_windows_f):
    #             denoised_frame_latent = denoised[:, :, [frame_idx]]
    #             video_frames_tensor = self.pretrained_t2v.decode_first_stage_2DAE(denoised_frame_latent)
    #             video_frames_list.append(video_frames_tensor)
    #
    #         videos = torch.cat(video_frames_list, dim=2)
    #         videos_chunked = torch.chunk(videos, denoised_chunk_num+2, dim=4)
    #
    #         videos = torch.cat(videos_chunked[1:-1], dim=4)
    #
    #     else:
    #         videos = denoised
    #
    #     return videos, denoised
    #


