from datetime import datetime
import os
import shutil

import argparse
from dataclasses import dataclass, field

@dataclass
class VArgs:
    seed: int = 8888
    gpu_id: int = 5
    loop_window_from_percentage_w: float = None
    merge_renoised_overlap_latent_ratio: float = 0.0
    merge_prev_step: int = 0
    

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        for field_name, field_def in cls.__dataclass_fields__.items():
            parser.add_argument(
                f'--{field_name}',
                type=type(field_def.default),
                default=field_def.default,
                help=f'{field_name} (default: {field_def.default})'
            )
        args = parser.parse_args()
        return cls(**vars(args))


vargs = VArgs.from_args()
print(vargs)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{vargs.gpu_id}"
os.environ["WORLD_SIZE"] = "1"

from utils.diffusion_utils import expand_per_frame_prompts, resize_video_latent
from utils.log_utils import CustomLog
from utils.loop_merge_utils import save_decoded_video_latents


from pipeline.vc2_lvdm_scheduler import lvdm_DDIM_Scheduler
from utils.precast_latent_utils import encode_images_list_to_latent_tensor, get_img_list_from_folder

from pipeline.t2v_vc2_EX_pipeline import VC2_Pipeline_T2V_EX

import torch
import imageio

from dataclasses import dataclass, field
from typing import Optional

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from scripts.evaluation.funcs import load_model_checkpoint, save_gif

from utils.utils import instantiate_from_config, set_directory

@dataclass
class RunArgs:
    config: str = "configs/inference_t2v_512_v2.0.yaml"
    base_ckpt_path: str = "./videocrafter_models/i2v_512_v1/model.ckpt"
    seed: int = 114514
    frame_length: int = 12              # f in paper
    num_partitions: int = 4             # n in paper
    num_inference_steps: int = None
    total_video_length: int = 64       # N in paper; desired length of the output video
    num_processes: int = 1
    rank: int = 0
    height: int = 320
    width: int = 512
    save_frames: bool = True
    fps: int = 8
    unconditional_guidance_scale: float = 7.5
    lookahead_denoising: bool = False
    eta: float = 1.0
    output_dir: Optional[str] = None
    use_mp4: bool = True
    # output_fps: int = 10

def main(run_args: RunArgs, prompt, # image_path, image_folder,
         dummy_prefix_frame=0,
         prefix_sampling_step=0,
         skip_num_partition=0,
         use_moving_img_cond=False,
         shift_moving_img_cond_by_rank=False,
         use_fp16=False, save_latents=False,

         loop_step=None,
         # num_windows_h=None,
         # num_windows_w=None,
         # num_windows_f=None,
         total_w=None,
         total_h=None,
         total_f=None,

         use_skip_time=False,
         skip_time_step_idx=0,
         progressive_skip=False,

         use_pre_denoise=False,
         pre_denoise_steps=None,
         skip_steps_after_pre_denoise=None,

         # shift_jump_odd_w=None,
         # shift_jump_odd_h=None,
         # shift_jump_odd_f=None,

         dock_at_h=None,
         dock_at_w=None,
         dock_at_f=None,
         docking_step_range=None,

         # merge_predenoise_ratio_list=None,
         merge_renoised_overlap_latent_ratio=None,
         merge_prev_denoised_ratio_list=None,

         per_frame_prompt_dict=None,

         random_shuffle_init_frame_stride=0,

         overlap_ratio_list=None,
         merge_overlap_ratio_list=None,
         init_stage_loop_step=None,
         loop_window_from_percentage_w=None,

         project_name="",
         project_folder=None):

    seed_everything(run_args.seed)

    output_dir, latents_dir = set_directory(project_id=project_name, project_folder=project_folder)

    custom_log = CustomLog()
    custom_log.log_to_file_and_terminal(os.path.join(output_dir, "log.txt"))

    source_file_path = __file__
    destination_file_path = os.path.join(output_dir, "_src_script.py")
    shutil.copy(source_file_path, destination_file_path)

    src_path = os.path.join(output_dir, 'src')
    os.makedirs(src_path)

    src_dirs_list = ["./utils", "./pipeline"]

    for src_dir in src_dirs_list:
        src_dir_abs = os.path.abspath(src_dir)
        target_save_dir = os.path.join(src_path, src_dir)
        target_save_dir_abs = os.path.abspath(target_save_dir)
        shutil.copytree(src_dir_abs, target_save_dir_abs, ignore=shutil.ignore_patterns('*.pyc'))

    # step 1: model config
    config = OmegaConf.load(run_args.config)
    model_config = config.pop("model", OmegaConf.create())

    if use_fp16:    # TODO :  有待测试
        model_config['params']['unet_config']['params']['use_fp16'] = True

    model = instantiate_from_config(model_config)
    model = model.cuda()
    assert os.path.exists(run_args.base_ckpt_path), f"Error: checkpoint [{run_args.base_ckpt_path}] Not Found!"

    model = load_model_checkpoint(model, run_args.base_ckpt_path)
    model.eval()

    if use_fp16:
        model.to(torch.float16)


    scheduler = lvdm_DDIM_Scheduler(model=model)

    pipeline = VC2_Pipeline_T2V_EX(pretrained_t2v=model,
                                   scheduler=scheduler,
                                   model_config=model_config)
    pipeline.to(model.device)

    if use_fp16:
        pipeline.to(model.device, torch_dtype=torch.float16)

    # sample shape
    assert (run_args.height % 16 == 0) and (run_args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # latent noise shape
    # h, w = run_args.height // 8, run_args.width // 8
    # frames = run_args.video_length
    channels = model.channels

    batch_size = 1
    prompts = [prompt]

    # basic_SW_video_frames, basic_SW_latent = pipeline.basic_sample_shift_multi_windows(
    #     prompt=prompts,
    #     # img_cond_path=img_cond_path,
    #     height=run_args.height,
    #     width=run_args.width,
    #     frames=16, # run_args.num_inference_steps,
    #     fps=run_args.fps,
    #     guidance_scale=run_args.unconditional_guidance_scale,

    #     init_panorama_latent=None,
    #     clear_pre_denoised_latent=None,
    #     use_skip_time=use_skip_time,
    #     skip_time_step_idx=skip_time_step_idx,
    #     progressive_skip=progressive_skip,
    #     num_windows_h=num_windows_h,
    #     num_windows_w=num_windows_w,
    #     num_windows_f=num_windows_f,
    #     loop_step=init_stage_loop_step,
    #     # pano_image_path=pano_image_path,

    #     use_pre_denoise=use_pre_denoise,
    #     pre_denoise_steps=pre_denoise_steps,
    #     skip_steps_after_pre_denoise=skip_steps_after_pre_denoise,

    #     # shift_jump_odd_w=shift_jump_odd_w,
    #     # shift_jump_odd_h=shift_jump_odd_h,
    #     # shift_jump_odd_f=shift_jump_odd_f,

    #     docking_w=False,
    #     docking_h=False,
    #     docking_f=False,
    #     docking_step_range=docking_step_range,

    #     merge_predenoise_ratio_list=None,

    #     random_shuffle_init_frame_stride=random_shuffle_init_frame_stride,

    #     latents=None,
    #     num_inference_steps=run_args.num_inference_steps,
    #     num_videos_per_prompt=1,
    #     generator_seed=run_args.seed,
    # )

    # # torch.save(basic_SW_latent, os.path.join(output_dir, f"basic_SW_latent_{project_name}.pt"))
    # # torch.save(basic_SW_video_frames, os.path.join(output_dir, f"basic_SW_video_frames_{project_name}.pt"))

    # interpolated_basic_SW_latent = resize_video_latent(input_latent=basic_SW_latent.clone(), mode="bicubic",
    #                                                    target_height=total_h // pipeline.vae_scale_factor,
    #                                                    target_width=total_w // pipeline.vae_scale_factor)

    # renoised_basic_SW_latent = pipeline._add_noise(interpolated_basic_SW_latent, run_args.num_inference_steps - 1)

    # save_decoded_video_latents(decoded_video_latents=basic_SW_video_frames,
    #                            output_path=output_dir,
    #                            output_name=f"basic_sw",
    #                            fps=run_args.fps)

    print("==== EX - UP Sampling ====")
    up_sampled_video_frames, up_sampled_latent = pipeline.basic_sample_shift_multi_windows_with_overlap(
        prompt=prompts,
        height=run_args.height,
        width=run_args.width,
        frames=16,
        fps=run_args.fps,
        guidance_scale=run_args.unconditional_guidance_scale,
        num_inference_steps=run_args.num_inference_steps,
        num_videos_per_prompt=1,
        generator_seed=run_args.seed,

        init_panorama_latent=None,

        total_w=total_w,
        total_h=total_h,
        total_f=total_f,

        dock_at_h=dock_at_h,
        dock_at_w=dock_at_w,
        dock_at_f=dock_at_f,

        loop_step=loop_step,

        loop_window_from_percentage_w=loop_window_from_percentage_w,

        merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,
        merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,
        per_frame_prompt_dict=per_frame_prompt_dict,

        overlap_ratio_list=overlap_ratio_list,
        init_stage_loop_step=init_stage_loop_step,
        merge_overlap_ratio_list=merge_overlap_ratio_list,
    )
    torch.save(up_sampled_latent, os.path.join(output_dir, f"up_sampled_latent_{project_name}.pt"))
    save_decoded_video_latents(decoded_video_latents=up_sampled_video_frames,
                               output_path=output_dir,
                               output_name=f"ex_{upscale_factor}X",
                               fps=run_args.fps)



if __name__ == "__main__":

    run_args = RunArgs()
    run_args.base_ckpt_path="./videocrafter_models/base_512_v2/model.ckpt"
    run_args.num_inference_steps = run_args.num_partitions * run_args.frame_length

    run_args.seed = vargs.seed

    dummy_prefix_frame = 0
    # 在 fifo 的起始阶段重复 dummy_prefix_frame 次第 0 张图片 ( window_0.png )

    run_args.fps = 8
    run_args.total_video_length = 512 + dummy_prefix_frame


    skip_num_partition = 0
    # 不完全加噪 (相当于跳过 run_args.num_partitions / skip_num_partition 比例的加噪, 值越大跳过越多)

    use_moving_img_cond = True
    shift_moving_img_cond_by_rank = True    # 调整 fifo 内层循环(rank) 中每次去噪使用 start_idx 对应的 原图 作为 image cond

    prefix_sampling_step = run_args.frame_length * (run_args.num_partitions - skip_num_partition) # run_args.num_inference_steps
    # 用 base ddim sample 初始化 fifo latent queue

    # prompt = "mountains at twilight, cloud moving" #
    # prompt = "rock band performing live on stage with colorful lights illuminating the crowd, audience cheers and dances"  # "fireworks burst in the sky above city skyline at night"

    prompt = "a group of hot air balloons flying over a green field under cloudy blue sky"
    # "Soft sakura blossom petals drift slowly in wind, falling onto a lake, creating gentle ripples across the water, surrounded by blooming sakura trees and tranquil, natural beauty"
    # "An astronaut walking on the moon’s surface, high-quality, 4K resolution"
    # "In a soccer match, players wearing red and blue uniforms running across a vibrant green field"
    # "A majestic waterfall in misty atmosphere, massive water column plunging down" # with misty spray and rainbows"
    # "A majestic waterfall cascading down from a towering cliff, massive water volume plunging three thousand feet into a misty gorge"
    # "river waters flow from left to right, carrying a small fishing boat drifting along with the current"

    # "a traveler walking through a desert landscape under the blazing sun and blue sky"
    # "green northern lights swirling and shifting across the night sky, reflected in the calm lake waters of a snow-covered landscape"
    # "An astronaut walking on the moon’s surface, high-quality, 4K resolution"
    # "A majestic bald eagle soaring through a clear blue sky, wings spread wide, against a backdrop of snow-capped mountains"
    # "Rushing river waters flow with powerful currents as fishing boats drift steadily downstream. The lush, green riverbanks are lined with dense foliage"
    #
    # "An astronaut walking on the moon’s surface, high-quality, 4K resolution"
    #
    # "A majestic waterfall cascades powerfully over a steep cliff, its waters plunging into the pool below"
    # "a view of fireworks exploding in the night sky over a city, as seen from a plane"
    # "a large mountain lake, the lake surrounded by hills and mountains"
    # "the top of a snow covered mountain range, with the sun shining over it"
    # "a group of hot air balloons flying over a green field under cloudy blue sky"
    #
    # "green northern lights swirling and shifting across the night sky, reflected in the calm lake waters of a snow-covered landscape"
    # "A rocket launch and fly up into the sky at dusk, surrounded by a massive plume of fiery smoke and steam"
    # "a large mountain lake, the lake surrounded by hills and mountains" # "fireworks burst in the sky above city skyline at night"

    image_folder_name = None

    image_path = None
    image_folder = None

    # pano_image_path = "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/input_img/city_fireworks_640x1024.png"

    # loop_step = 16

    num_windows_w = 2
    num_windows_h = 1
    num_windows_f = 1

    upscale_factor = 2

    total_w = 512 * upscale_factor * num_windows_w
    total_h = 320 * upscale_factor * num_windows_h
    total_f = 16

    use_skip_time = False
    skip_time_step_idx = 0

    progressive_skip = False            # 从第 0 帧开始逐渐减增大噪声, 直到第 skip_time_step_idx 帧完全是噪声, 按照不 skip 的 time step 来去噪

    use_pre_denoise = False              # 预先做一次 naive 去噪, 然后放大到 num_window 倍, 再用 SW
    pre_denoise_steps = 48
    skip_steps_after_pre_denoise = 0    # 在 SW 去噪过程的 ts 跳过中抵消 skip_steps_after_pre_denoise 步的 skip_time_step_idx,
                                        # 但是 加噪仍然按照 progressive_skip, 也就是非0值将使 SW 按照高噪声的标准去噪实际上低噪声的 latent
                                        # 可能用于缓解 spatial SW (t2v) 中的 上下分裂/虚影问题


    dock_at_h = True                    # 阻止 window 跨越边界, 而是改为 docking 在边界, 将会增加 T 次去噪步数
    dock_at_w = True
    dock_at_f = True

    loop_window_from_percentage_w = vargs.loop_window_from_percentage_w

    docking_step_range = range(0, 20)       # 使用 docking 的 step_index (i) 范围

    # random_shuffle_init_frame_stride = 0.5

    # merge_predenoise_ratio_list = [1 - ((1+math.cos(t*math.pi))/2) ** _residual_scale_factor
    #                                for t in [t/run_args.num_inference_steps for t in range(run_args.num_inference_steps)]
    #                                ]
    # merge_predenoise_ratio_list[10:] = [1.0] * len(merge_predenoise_ratio_list[10:])
    #
    # print(f"merge_predenoise_ratio_list: {merge_predenoise_ratio_list}")
    #
    #                                         # 类似 Demo Diffusion 的 Latent Skip Residual,
    #                                         # 将加入对应步数的噪声后的 resized_latent 与当前去噪得到的 Pano Latent 加权平均
    #                                         # list 中每一项对应 i 步的 curr latent 保留比例, 越大 resized_latent 注入的越少


    #
    # _residual_step = 10
    # _residual_linear_max_ratio = 0.1
    # merge_overlap_ratio_list = [_residual_linear_max_ratio * t/_residual_step for t in range(_residual_step)] + [1.0] * (run_args.num_inference_steps-_residual_step)
    # print(f"merge_predenoise_ratio_list: {merge_overlap_ratio_list}")

    init_stage_loop_step = 8

    overlap_ratio_list_0 = [i / 4 for i in range(4)]
    overlap_ratio_list = [overlap_ratio_list_0[-1 - i // (run_args.num_inference_steps // len(overlap_ratio_list_0))] for i in
                          range(run_args.num_inference_steps)]

    print(f"overlap_ratio_list: {overlap_ratio_list}")

    merge_renoised_overlap_latent_ratio = vargs.merge_renoised_overlap_latent_ratio       # 加权平均 overlap 部分的去噪结果, 值越大代表 prev 的结果占比越大
    _merge_prev_step = vargs.merge_prev_step
    merge_prev_denoised_ratio_list = [merge_renoised_overlap_latent_ratio * (1 - t / _merge_prev_step) for t in range(_merge_prev_step)] + [0] * (run_args.num_inference_steps - _merge_prev_step)
    # 加权平均 overlap 部分的去噪结果
    print(f"merge_prev_denoised_ratio_list: {merge_prev_denoised_ratio_list}")



    PROJECT_FOLDER = "T2V_EX_CASEs-noGMG"
    PROJECT_NOTE = f"T2V_SW_-EX-overlap_VAR{merge_renoised_overlap_latent_ratio}-{_merge_prev_step}"
    PROJECT_NAME = f"{PROJECT_NOTE}" # \

    save_latents = False

    main(run_args, prompt, # image_path, image_folder,
         dummy_prefix_frame=dummy_prefix_frame,
         prefix_sampling_step=prefix_sampling_step,
         skip_num_partition=skip_num_partition,
         use_moving_img_cond=use_moving_img_cond,
         shift_moving_img_cond_by_rank=shift_moving_img_cond_by_rank,
         project_name=PROJECT_NAME,
         project_folder=PROJECT_FOLDER,

         # pano_image_path=pano_image_path,
         # loop_step=loop_step,
         total_w=total_w,
         total_h=total_h,
         total_f=total_f,

         use_skip_time=use_skip_time,
         skip_time_step_idx=skip_time_step_idx,
         progressive_skip=progressive_skip,

         use_pre_denoise=use_pre_denoise,
         pre_denoise_steps=pre_denoise_steps,
         skip_steps_after_pre_denoise=skip_steps_after_pre_denoise,


         dock_at_h=dock_at_h,
         dock_at_w=dock_at_w,
         dock_at_f=dock_at_f,
         docking_step_range=docking_step_range,

         overlap_ratio_list=overlap_ratio_list,
         # merge_overlap_ratio_list=merge_overlap_ratio_list,
         init_stage_loop_step=init_stage_loop_step,
         merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,
         merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

         # per_frame_prompt_dict=per_frame_prompt_dict,

         loop_window_from_percentage_w=loop_window_from_percentage_w,

         save_latents=save_latents)
