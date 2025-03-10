import os
import random
import time
import warnings

import cv2
import math
import numpy as np
import torch
from PIL import Image

from tqdm import trange
from typing import List, Optional, Union, Dict, Any
from diffusers import logging

from diffusers import DiffusionPipeline
from lvdm.models.ddpm3d import LatentDiffusion
from pipeline.t2v_vc2_pipeline import VC2_Pipeline_T2V
from pipeline.vc2_lvdm_scheduler import lvdm_DDIM_Scheduler
from utils.diffusion_utils import resize_video_latent
from utils.precast_latent_utils import encode_images_list_to_latent_tensor
from utils.shift_window_utils import RingLatent
from utils.tensor_utils import mix_latents_with_mask

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class VC2_Pipeline_T2V_EX(VC2_Pipeline_T2V):

    @torch.no_grad()
    def basic_sample_time_expand(
            self,
            prompt: Union[str, List[str]] = None,
            # img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # TODO: 已给出部分的 noise 也应该保持与生产时使用的相同 ?

            init_panorama_latent: torch.Tensor = None,  # 包含整个 panorama 的 latent
            # clear_pre_denoised_latent: torch.Tensor = None,

            num_windows_w: int = None,                  # 总宽度是 width(512) * num_windows_w
            num_windows_h: int = None,                  # 总高度是 height(320) * num_windows_h
            num_windows_f: int = None,
            loop_step: int = None,                      # 应大于1, 越小 window 滑动越快      # doubt: 是否应该也分 h / w

            multi_prompt_dict: dict = None,

            random_shuffle_init_frame_stride=0,         # 同 FreeNoise 中的 Noise Rescheduling (shuffle 第一个 frame_window 的 noise 以替代后面的 init noise)

            overlap_ratio_list:List[float] = None,      # 重叠部分的比例
            init_stage_loop_step:int = None,    #

            # merge_overlap_ratio_list=None,
            merge_renoised_overlap_latent_ratio:float = 1,
            merge_prev_denoised_ratio_list: list[float] = None,

            per_frame_prompt_dict: dict = None,

            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            **kwargs
    ):
        assert num_windows_h==1 and num_windows_w==1, "not Implemented for HW up-scaled time expand now"

        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)
        cond = {"c_crossattn": [text_emb], "fps": fps}


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

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)  # set_timesteps(num_inference_steps)   # , lcm_origin_steps)

        full_timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]

        # if use_skip_time and not progressive_skip:
        #     timesteps = full_timesteps[skip_time_step_idx-skip_steps_after_pre_denoise:]
        #     print(f"skip : {skip_time_step_idx}")
        # else:
        #     timesteps = full_timesteps

        timesteps = full_timesteps


        print(f"[basic_sample_time_expand] denoise timesteps: {timesteps}")
        # print(f"[basic_sample_time_expand] SKIP {skip_time_step_idx}-{skip_steps_after_pre_denoise} = {skip_time_step_idx-skip_steps_after_pre_denoise} timesteps {'(progressive)' if progressive_skip else ''}")
        # print(f"[basic_sample_time_expand] skip_steps_after_pre_denoise = {skip_steps_after_pre_denoise}")

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
        bs = batch_size * num_videos_per_prompt  # ?

        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

        else:
            print("[basic_sample_time_expand] using given init latent")

            assert init_panorama_latent.shape == total_shape, f"[basic_sample_time_expand] " \
                                                              f"init_panorama_latent shape {init_panorama_latent.shape}" \
                                                              f"does not match" \
                                                              f"desired shape {total_shape}"
            init_panorama_latent = init_panorama_latent.clone()
            raise NotImplementedError



        panorama_ring_latent_handler = RingLatent(init_latent=init_panorama_latent)
        panorama_ring_latent_denoised_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))

        # define window shift




        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:

            for i, t in enumerate(timesteps):
                print(f"\n i = {i}")

                overlap_ratio = overlap_ratio_list[i]

                total_window_num_f = math.ceil((num_windows_f - 1) / (1 - overlap_ratio)) + 1

                # shift overlap window, 每个 overlap window 整体移动
                offset_shift_step_size_f = max(int( overlap_ratio * frames / init_stage_loop_step), 1)
                # TODO: check, 这里为了测试强行设置了最小值为 1

                latent_frames_begin = (i % init_stage_loop_step) * offset_shift_step_size_f

                # prev_window_latent_frame_begin = -1


                # reset denoised mask record
                panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))


                # for shift_f_idx in sparse_denoise_list

                # for shift_f_idx in [-100] + list(range(total_window_num_f)) + [-101]:

                additional_denoise_num = int(overlap_ratio*total_window_num_f)
                additional_denoise_idx_list = list(range(total_window_num_f, total_window_num_f + additional_denoise_num))
                curr_shift_f_idies_list = list(range(total_window_num_f)) + additional_denoise_idx_list

                # TODO: 增多这种过长"回头"去噪的frame长度 具体设置有待考量

                print(f"curr overlap ratio = {overlap_ratio}, additional_denoise_num = {additional_denoise_num}, additional_denoise_idx_list = {additional_denoise_idx_list}")

                for shift_f_idx in curr_shift_f_idies_list:

                    window_latent_left = 0 # latent_pos_left_start + shift_w_idx * latent_width
                    window_latent_right = window_latent_left + latent_width
                    window_latent_top = 0 # latent_pos_top_start + shift_h_idx * latent_height
                    window_latent_down = window_latent_top + latent_height

                    window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio))
                    window_latent_frame_begin = window_latent_frame_begin % (frames * num_windows_f)
                    window_latent_frame_end = window_latent_frame_begin + frames

                    if shift_f_idx == -777:
                        if i == 0:      # Notes: 在 i = 0 时 window 是恰好对齐 frames 数量的, 故增加一个平滑过渡机制,
                                        #   也可以考虑改为算 latent_frames_begin 时 i 增加一个 offset
                                        #   TODO: 增多这种过长"回头"去噪的frame长度
                            shift_f_idx = total_window_num_f
                            window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio))
                            window_latent_frame_end = window_latent_frame_begin + frames
                        else:
                            continue

                    curr_prompt = prompt
                    if per_frame_prompt_dict is not None:
                        curr_prompt = per_frame_prompt_dict[window_latent_frame_begin]
                        text_emb = self.pretrained_t2v.get_learned_conditioning([curr_prompt])
                        cond = {"c_crossattn": [text_emb], "fps": fps}


                    # if shift_f_idx == -100:  # dock at begin
                    #     if latent_frames_begin == 0:
                    #         print(
                    #             f"shift_f_idx = {shift_f_idx}: latent_frames_begin is already 0, skipped dock padding")
                    #         continue
                    #     else:
                    #         window_latent_frame_begin = 0
                    #         window_latent_frame_end = frames
                    # elif shift_f_idx == -101:  # dock at end 补齐最后的空缺
                    #     window_latent_frame_begin = num_windows_f * frames - frames
                    #     window_latent_frame_end = num_windows_f * frames
                    # else:
                    #     window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio))
                    #     window_latent_frame_end = window_latent_frame_begin + frames

                    # if window_latent_frame_end > num_windows_f * frames:
                    #     print(f"shift_f_idx = {shift_f_idx}: window_latent_frame_end {window_latent_frame_end} > total_frames {frames * num_windows_f}]")
                    #     continue

                    window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                                                                                   pos_right=window_latent_right,
                                                                                   pos_top=window_latent_top,
                                                                                   pos_down=window_latent_down,
                                                                                   frame_begin=window_latent_frame_begin,
                                                                                   frame_end=window_latent_frame_end)

                    window_latent_prev_denoise = window_latent.clone()

                    # # temporal 的 skip residual ( 加噪部分 ) # 甚至可能可以考虑省略加噪    # Notes: 已被 0/1 mask 取代
                    # if prev_window_latent_frame_begin != -1 and i < total_steps - 1 and merge_overlap_ratio_list is not None and overlap_ratio > 0:
                    #
                    #     prev_window_latent_frame_end = prev_window_latent_frame_begin + frames
                    #
                    #     overlap_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                    #                                                                     pos_right=window_latent_right,
                    #                                                                     pos_top=window_latent_top,
                    #                                                                     pos_down=window_latent_down,
                    #                                                                     frame_begin=window_latent_frame_begin,
                    #                                                                     frame_end=prev_window_latent_frame_end)
                    #
                    #     renoised_overlap_latent = self.scheduler.re_noise(x_a=overlap_latent.clone(),
                    #                                                       step_a=total_steps - i - 1 - 1,   # next step
                    #                                                       step_b=total_steps - i - 1)       # curr_step
                    #
                    #     window_latent[:, :, 0:prev_window_latent_frame_end-window_latent_frame_begin, :, :] = renoised_overlap_latent.clone()
                    #
                    #     print(f"temporal skip residual: renoising [{window_latent_frame_begin}:{prev_window_latent_frame_end}] "
                    #           f"-> window_latent[0:{prev_window_latent_frame_end-window_latent_frame_begin}]")

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
                        # window_denoised_mask = window_denoised_mask[0, 0, [0]]    # Notes: f 不应该被截取, TODO: 修改其他地方的 mask 机制以支持 f 拓展
                        window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                              latent_to_add=noised_window_latent,
                                                              mask=window_denoised_mask,
                                                              mix_ratio=1)

                    print(f"shift_f_idx = {shift_f_idx} => [{window_latent_frame_begin} - {window_latent_frame_end}], curr prompt: {curr_prompt}")

                    window_latent, denoised = self._basic_denoise_one_step(t=t, i=i, total_steps=total_steps,
                                                                           device=device, latent=window_latent,
                                                                           cond=cond, uncond=uncond,
                                                                           guidance_scale=guidance_scale,
                                                                           frames=frames, bs=bs)

                    if merge_prev_denoised_ratio_list is not None and i < total_steps - 1:     # Notes: 加权平均 overlap 部分的去噪结果

                        merge_prev_denoised_ratio = merge_prev_denoised_ratio_list[i]
                        window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                              latent_to_add=window_latent_prev_denoise,
                                                              mask=window_denoised_mask,
                                                              mix_ratio=merge_prev_denoised_ratio)        # TODO: 考虑更精细的比例控制方式

                    # # temporal 的 skip residual ( 加噪部分 )
                    # if prev_window_latent_frame_begin != -1 and i < total_steps - 1 and merge_overlap_ratio_list is not None and overlap_ratio > 0:
                    #
                    #     curr_merge_ratio = merge_overlap_ratio_list[i]
                    #     print(f"merging residual latent: {round(curr_merge_ratio, 3)} * curr + {round(1.0 - curr_merge_ratio, 3)} * noised_resized")
                    #
                    #     curr_overlap_latent = window_latent[:, :, 0:prev_window_latent_frame_end-window_latent_frame_begin, :, :]
                    #     mixed_residual_latent = curr_merge_ratio * curr_overlap_latent + (1.0 - curr_merge_ratio) * overlap_latent
                    #
                    #     window_latent[:, :, 0:prev_window_latent_frame_end-window_latent_frame_begin, :, :] = mixed_residual_latent.clone()


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

                    new_window_denoised_mask = torch.ones_like(window_latent, dtype=window_latent.dtype,
                                                               device=window_latent.device)
                    panorama_ring_mask_handler.set_window_latent(new_window_denoised_mask,
                                                                 pos_left=window_latent_left,
                                                                 pos_right=window_latent_right,
                                                                 pos_top=window_latent_top,
                                                                 pos_down=window_latent_down,
                                                                 frame_begin=window_latent_frame_begin,
                                                                 frame_end=window_latent_frame_end)
                    # prev_window_latent_frame_begin = window_latent_frame_begin

                progress_bar.update()

        denoised = panorama_ring_latent_denoised_handler.torch_latent.clone().to(device=init_panorama_latent.device)

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)      # doubt: VAE 能正常 decode 超宽的 latent 吗 ?
        else:
            videos = denoised

        return videos, denoised

    @torch.no_grad()
    def basic_sample_shift_multi_windows_with_overlap(
            self,
            prompt: Union[str, List[str]] = None,
            # img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # TODO: 已给出部分的 noise 也应该保持与生产时使用的相同 ?

            init_panorama_latent: torch.Tensor = None,  # 包含整个 panorama 的 latent
            skip_timestep: int = None,  # 跳过的去噪步数
            # clear_pre_denoised_latent: torch.Tensor = None,

            total_w: int = None,                  # 总宽度是 width(512) * num_windows_w
            total_h: int = None,                  # 总高度是 height(320) * num_windows_h
            total_f: int = None,
            # loop_step: int = None,                      # 应大于1, 越小 window 滑动越快      # doubt: 是否应该也分 h / w

            dock_at_h=None,
            dock_at_w=None,   # TODO
            dock_at_f=None,

            # multi_prompt_dict: dict = None,   # TODO: 支持 multi_prompt (spatial + temporal)
            # per_frame_prompt_dict: dict = None,

            overlap_ratio_list:List[float] = None,      # 重叠部分的比例
            init_stage_loop_step:int = None,    #

            # additional_denoise_num_list:List[int] = None,   # 需要额外回环去噪的 window 数量

            merge_renoised_overlap_latent_ratio:float = 1,
            merge_prev_denoised_ratio_list: list[float] = None,

            loop_window_from_percentage_h: float = None,    # 从 idies_list 的X%开始循环
            loop_window_from_percentage_w: float = None,
            loop_window_from_percentage_f: float = None,


            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            **kwargs
    ):
        # assert num_windows_h==1 and num_windows_w==1, "not Implemented for HW up-scaled time expand now"

        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)
        cond = {"c_crossattn": [text_emb], "fps": fps}


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

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)  # set_timesteps(num_inference_steps)   # , lcm_origin_steps)

        full_timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]

        # if use_skip_time and not progressive_skip:
        #     timesteps = full_timesteps[skip_time_step_idx-skip_steps_after_pre_denoise:]
        #     print(f"skip : {skip_time_step_idx}")
        # else:
        #     timesteps = full_timesteps
        if skip_timestep is not None:
            assert init_panorama_latent is not None, "skip_timestep should use with input init latent"
            timesteps = full_timesteps[skip_timestep:]
        else:
            timesteps = full_timesteps


        print(f"[basic_sample_time_expand] denoise timesteps: {timesteps}")
        # print(f"[basic_sample_time_expand] SKIP {skip_time_step_idx}-{skip_steps_after_pre_denoise} = {skip_time_step_idx-skip_steps_after_pre_denoise} timesteps {'(progressive)' if progressive_skip else ''}")
        # print(f"[basic_sample_time_expand] skip_steps_after_pre_denoise = {skip_steps_after_pre_denoise}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable [pano]
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
        bs = batch_size * num_videos_per_prompt  # ?

        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

        else:
            print("[basic_sample_time_expand] using given init latent")

            assert init_panorama_latent.shape == total_shape, f"[basic_sample_time_expand] " \
                                                              f"init_panorama_latent shape {init_panorama_latent.shape}" \
                                                              f"does not match" \
                                                              f"desired shape {total_shape}"
            init_panorama_latent = init_panorama_latent.clone()
            # raise NotImplementedError



        panorama_ring_latent_handler = RingLatent(init_latent=init_panorama_latent)
        panorama_ring_latent_denoised_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))

        # define window shift

        _DOCK_START_INDEX = -101
        _DOCK_END_INDEX = -111


        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:

            for i, t in enumerate(timesteps):
                print(f"\n i = {i}")


                overlap_ratio = overlap_ratio_list[i]

                total_window_num_h = math.ceil((total_h // height - 1) / (1 - overlap_ratio)) + 1
                total_window_num_w = math.ceil((total_w // width - 1) / (1 - overlap_ratio)) + 1
                total_window_num_f = math.ceil((total_f // frames - 1) / (1 - overlap_ratio)) + 1

                # shift overlap window, 每个 overlap window 整体移动
                if total_h > height:
                    offset_shift_step_size_h = max(int(overlap_ratio * latent_height / init_stage_loop_step), 1)
                    latent_pos_top_start = (i % init_stage_loop_step) * offset_shift_step_size_h
                    # TODO: additional denoise
                    curr_shift_h_idies_list = list(range(total_window_num_h))

                    if dock_at_h:
                        curr_shift_h_idies_list = [_DOCK_START_INDEX] + curr_shift_h_idies_list + [_DOCK_END_INDEX]

                    if loop_window_from_percentage_h is not None:
                        new_start_index_h = int(loop_window_from_percentage_h * len(curr_shift_h_idies_list)-1)
                        curr_shift_h_idies_list = curr_shift_h_idies_list[new_start_index_h:] + curr_shift_h_idies_list[:new_start_index_h]

                elif total_h == height:
                    latent_pos_top_start = 0
                    curr_shift_h_idies_list = [0]
                else:
                    print(f"total_w {total_w} should >= width {width} !")
                    raise ValueError

                if total_w > width:
                    offset_shift_step_size_w = max(int(overlap_ratio * latent_width / init_stage_loop_step), 1)
                    latent_pos_left_start = (i % init_stage_loop_step) * offset_shift_step_size_w
                    # TODO: additional denoise
                    curr_shift_w_idies_list = list(range(total_window_num_w))

                    if dock_at_w:
                        curr_shift_w_idies_list = [_DOCK_START_INDEX] + curr_shift_w_idies_list + [_DOCK_END_INDEX]

                    if loop_window_from_percentage_w is not None:
                        new_start_index_w = int(loop_window_from_percentage_w * len(curr_shift_h_idies_list)-1)
                        curr_shift_w_idies_list = curr_shift_w_idies_list[new_start_index_w:] + curr_shift_w_idies_list[:new_start_index_w]

                elif total_w == width:
                    latent_pos_left_start = 0
                    curr_shift_w_idies_list = [0]
                else:
                    print(f"total_w {total_w} should >= width {width} !")
                    raise ValueError

                # additional_denoise_num_f = 0
                # additional_denoise_idx_list_f = []
                if total_f > frames:    # 需要 shift on F

                    offset_shift_step_size_f = max(int(overlap_ratio * frames / init_stage_loop_step), 1)
                    # TODO: check, 这里为了测试强行设置了最小值为 1
                    latent_frames_begin = (i % init_stage_loop_step) * offset_shift_step_size_f
                    curr_shift_f_idies_list = list(range(total_window_num_f))
                    # if additional_denoise_num_list is not None:
                    #     additional_denoise_num_f = int(overlap_ratio * total_window_num_f)    # TODO: 增多这种过长"回头"去噪的frame长度 具体设置有待考量
                    #     additional_denoise_idx_list_f = list(range(total_window_num_f, total_window_num_f + additional_denoise_num_f))
                    #     curr_shift_f_idies_list = list(range(total_window_num_f)) + additional_denoise_idx_list_f

                    if dock_at_f:
                        curr_shift_f_idies_list = [_DOCK_START_INDEX] + curr_shift_f_idies_list + [_DOCK_END_INDEX]

                    if loop_window_from_percentage_f is not None:
                        new_start_index_f = int(loop_window_from_percentage_f*len(curr_shift_h_idies_list)-1)
                        curr_shift_f_idies_list = curr_shift_f_idies_list[new_start_index_f:] + curr_shift_f_idies_list[:new_start_index_f]


                elif total_f == frames: # 无需 shift on F
                    latent_frames_begin = 0
                    curr_shift_f_idies_list = [0]
                else:
                    print(f"total_f {total_f} should >= frames {frames} !")
                    raise ValueError

                # reset denoised mask record
                panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))


                # print(f"curr overlap ratio = {overlap_ratio}, additional_denoise_num = {additional_denoise_num}, additional_denoise_idx_list = {additional_denoise_idx_list}")

                for shift_h_idx in curr_shift_h_idies_list:

                    window_latent_top = latent_pos_top_start + shift_h_idx * int(latent_height * (1 - overlap_ratio))
                    window_latent_top = window_latent_top % (total_h // self.vae_scale_factor)
                    window_latent_down = window_latent_top + latent_height

                    if dock_at_h:
                        if shift_h_idx == _DOCK_START_INDEX:
                            if latent_pos_top_start == 0 and total_window_num_h > 1:
                                print(f"i % init_stage_loop_step = {i} % {init_stage_loop_step} = 0, no need for docking, skipped")
                                continue
                            window_latent_top = 0
                            window_latent_down = window_latent_top + latent_height

                        if shift_h_idx == _DOCK_END_INDEX:
                            if latent_pos_top_start == 0 and total_window_num_h > 1:
                                print(f"i % init_stage_loop_step = {i} % {init_stage_loop_step} = 0, no need for docking, skipped")
                                continue
                            window_latent_top = total_h // self.vae_scale_factor - latent_height
                            window_latent_down = window_latent_top + latent_height

                        if window_latent_down > total_h // self.vae_scale_factor:
                            print(f"window_latent_down = {window_latent_down} > down edge = {total_h // self.vae_scale_factor}, skipped because docking H")
                            continue

                    for shift_w_idx in curr_shift_w_idies_list:

                        window_latent_left = latent_pos_left_start + shift_w_idx * int(latent_width * (1 - overlap_ratio))
                        window_latent_left = window_latent_left % (total_w // self.vae_scale_factor)
                        window_latent_right = window_latent_left + latent_width

                        if dock_at_w:
                            if shift_w_idx == _DOCK_START_INDEX:
                                if latent_pos_left_start == 0 and total_window_num_w > 1:
                                    print(
                                        f"i % init_stage_loop_step = {i} % {init_stage_loop_step} = 0, no need for docking, skipped")
                                    continue
                                window_latent_left = 0
                                window_latent_right = window_latent_left + latent_width
                                window_image_left = 0
                                window_image_right = window_image_left + width

                            if shift_w_idx == _DOCK_END_INDEX :
                                if latent_pos_left_start == 0 and total_window_num_w > 1:
                                    print(
                                        f"i % init_stage_loop_step = {i} % {init_stage_loop_step} = 0, no need for docking, skipped")
                                    continue
                                window_latent_left = total_w // self.vae_scale_factor - latent_width
                                window_latent_right = window_latent_left + latent_width
                                window_image_left = total_w - width
                                window_image_right = window_image_left + width

                            if window_latent_right > total_w // self.vae_scale_factor:
                                print(
                                    f"window_latent_right = {window_latent_right} > right edge = {total_w // self.vae_scale_factor}, skipped because docking W")
                                continue

                            for shift_f_idx in curr_shift_f_idies_list:

                                window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio))
                                window_latent_frame_begin = window_latent_frame_begin % total_f
                                window_latent_frame_end = window_latent_frame_begin + frames
    
                                # todo: dock F

                                # if shift_f_idx == -777:
                                #     if i == 0:      # Notes: 在 i = 0 时 window 是恰好对齐 frames 数量的, 故增加一个平滑过渡机制,
                                #                     #   也可以考虑改为算 latent_frames_begin 时 i 增加一个 offset
                                #                     #   TODO: 增多这种过长"回头"去噪的frame长度
                                #         shift_f_idx = total_window_num_f
                                #         window_latent_frame_begin = latent_frames_begin + shift_f_idx * int(frames * (1 - overlap_ratio))
                                #         window_latent_frame_end = window_latent_frame_begin + frames
                                #     else:
                                #         continue
                                if shift_f_idx == _DOCK_START_INDEX:
                                    if latent_frames_begin == 0 and total_window_num_f > 1:
                                        print(
                                            f"i % init_stage_loop_step = {i} % {init_stage_loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_frame_begin = 0
                                    window_latent_frame_end = window_latent_frame_begin + frames

                                if shift_f_idx == _DOCK_END_INDEX:
                                    if latent_frames_begin == 0 and total_window_num_f > 1:
                                        print(
                                            f"i % init_stage_loop_step = {i} % {init_stage_loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_frame_begin = total_f - frames
                                    window_latent_frame_end = window_latent_frame_begin + frames

                                if window_latent_frame_end > total_f:
                                    print(
                                        f"window_latent_frame_end = {window_latent_frame_end} > total_f = {total_f}, skipped because docking F")
                                    continue

                                curr_prompt = prompt
                                # if per_frame_prompt_dict is not None:
                                #     curr_prompt = per_frame_prompt_dict[window_latent_frame_begin]
                                #     text_emb = self.pretrained_t2v.get_learned_conditioning([curr_prompt])
                                #     cond = {"c_crossattn": [text_emb], "fps": fps}


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
                                    # window_denoised_mask = window_denoised_mask[0, 0, [0]]    # Notes: f 不应该被截取, TODO: 修改其他地方的 mask 机制以支持 f 拓展
                                    window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                                          latent_to_add=noised_window_latent,
                                                                          mask=window_denoised_mask,
                                                                          mix_ratio=1)

                                print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
                                      f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
                                      f"")

                                window_latent, denoised = self._basic_denoise_one_step(t=t, i=i, total_steps=total_steps,
                                                                                       device=device, latent=window_latent,
                                                                                       cond=cond, uncond=uncond,
                                                                                       guidance_scale=guidance_scale,
                                                                                       frames=frames, bs=bs)

                                if merge_prev_denoised_ratio_list is not None and i < total_steps - 1:     # Notes: 加权平均 overlap 部分的去噪结果

                                    merge_prev_denoised_ratio = merge_prev_denoised_ratio_list[i]
                                    window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                                          latent_to_add=window_latent_prev_denoise,
                                                                          mask=window_denoised_mask,
                                                                          mix_ratio=merge_prev_denoised_ratio)        # TODO: 考虑更精细的比例控制方式

                                # # temporal 的 skip residual ( 加噪部分 )
                                # if prev_window_latent_frame_begin != -1 and i < total_steps - 1 and merge_overlap_ratio_list is not None and overlap_ratio > 0:
                                #
                                #     curr_merge_ratio = merge_overlap_ratio_list[i]
                                #     print(f"merging residual latent: {round(curr_merge_ratio, 3)} * curr + {round(1.0 - curr_merge_ratio, 3)} * noised_resized")
                                #
                                #     curr_overlap_latent = window_latent[:, :, 0:prev_window_latent_frame_end-window_latent_frame_begin, :, :]
                                #     mixed_residual_latent = curr_merge_ratio * curr_overlap_latent + (1.0 - curr_merge_ratio) * overlap_latent
                                #
                                #     window_latent[:, :, 0:prev_window_latent_frame_end-window_latent_frame_begin, :, :] = mixed_residual_latent.clone()


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

                                new_window_denoised_mask = torch.ones_like(window_latent, dtype=window_latent.dtype,
                                                                           device=window_latent.device)
                                panorama_ring_mask_handler.set_window_latent(new_window_denoised_mask,
                                                                             pos_left=window_latent_left,
                                                                             pos_right=window_latent_right,
                                                                             pos_top=window_latent_top,
                                                                             pos_down=window_latent_down,
                                                                             frame_begin=window_latent_frame_begin,
                                                                             frame_end=window_latent_frame_end)
                                # prev_window_latent_frame_begin = window_latent_frame_begin

                progress_bar.update()

        denoised = panorama_ring_latent_denoised_handler.torch_latent.clone().to(device=init_panorama_latent.device)

        if not output_type == "latent":
            denoised_chunk_num = 16
            denoised_chunked = list(torch.chunk(denoised, denoised_chunk_num, dim=4))
            denoised_chunked_cat_list = [denoised_chunked[-1]] + denoised_chunked + [denoised_chunked[0]]
            denoised = torch.cat(denoised_chunked_cat_list, dim=4)

            # videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)      # doubt: VAE 能正常 decode 超宽的 latent 吗 ?

            if not (total_w // width <= 2 and total_h // height <= 2):
                h_tile_num = min(total_h // height, 1)
                w_tile_num = total_w // width
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