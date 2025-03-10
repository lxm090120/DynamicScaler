from datetime import datetime
import os
import shutil


import argparse
from dataclasses import dataclass, field

@dataclass
class VArgs:
    seed:   int = 23333
    gpu_id: int = 6
    denoise_to_step:    int = 5
    skip_time_step:     int = 1
    loop_step_theta:    int = 10

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

from utils.log_utils import CustomLog
from utils.loop_merge_utils import save_decoded_video_latents
from pipeline.i2v_vc2_sphere_panorama_pipeline import VC2_Pipeline_I2V_SpherePano
from pipeline.vc2_lvdm_scheduler import lvdm_DDIM_Scheduler
from utils.precast_latent_utils import encode_images_list_to_latent_tensor, get_img_list_from_folder

import torch
import imageio

from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from scripts.evaluation.funcs import load_model_checkpoint, save_gif

from utils.utils import instantiate_from_config, set_directory

@dataclass
class RunArgs:
    config: str = "configs/inference_i2v_512_v1.0.yaml"
    base_ckpt_path: str = "./videocrafter_models/i2v_512_v1/model.ckpt"
    seed: int = 2333
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

def main(run_args: RunArgs, prompt, image_path, image_folder,
         dummy_prefix_frame=0,
         prefix_sampling_step=0,
         skip_num_partition=0,
         use_moving_img_cond=False,
         shift_moving_img_cond_by_rank=False,
         use_fp16=False, save_latents=False,

         pano_image_path=None,
         loop_step=None,
         num_windows_h=None,
         num_windows_w=None,
         num_windows_f=None,

         use_skip_time=False,
         skip_time_step_idx=0,
         progressive_skip=False,

         equirect_width=None,
         equirect_height=None,
         phi_theta_dict=None,
         phi_prompt_dict: dict = None,
         view_fov=None,
         loop_step_theta=None,
         merge_renoised_overlap_latent_ratio=None,
         paste_on_static=None,

         downsample_factor_before_vae_decode=None,
         view_get_scale_factor=None,
         view_set_scale_factor=None,

         denoise_to_step=None,

         num_windows_h_2=None,
         num_windows_w_2=None,

         project_name="",
         project_folder=None):

    print(f"==========================\n"
          f"CURR GPU: {os.environ['CUDA_VISIBLE_DEVICES']}, SEED: {run_args.seed}\n"
          f"==========================\n")

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

    # scheduler = DDIMScheduler(
    #     beta_start=model_config["params"]["linear_start"],
    #     beta_end=model_config["params"]["linear_end"],
    # )

    scheduler = lvdm_DDIM_Scheduler(model=model)

    pipeline = VC2_Pipeline_I2V_SpherePano(pretrained_t2v=model,
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
    img_cond_path = [pano_image_path]

    print("==== Sphere Panorama Shift Windows Sample ====")

    sphere_SW_latent, sphere_SW_denoised = pipeline.basic_sample_shift_shpere_panorama(
        prompt=prompts,
        img_cond_path=img_cond_path,
        height=run_args.height,
        width=run_args.width,
        frames=16, # run_args.num_inference_steps,
        fps=run_args.fps,
        guidance_scale=run_args.unconditional_guidance_scale,

        init_panorama_latent=None,
        use_skip_time=use_skip_time,
        skip_time_step_idx=skip_time_step_idx,
        progressive_skip=progressive_skip,
        num_windows_h=num_windows_h,
        num_windows_w=num_windows_w,
        num_windows_f=num_windows_f,
        loop_step=loop_step,
        pano_image_path=pano_image_path,

        equirect_width=equirect_width,
        equirect_height=equirect_height,
        phi_theta_dict=phi_theta_dict,
        phi_prompt_dict=phi_prompt_dict,
        view_fov=view_fov,
        loop_step_theta=loop_step_theta,
        merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,

        paste_on_static=paste_on_static,

        view_get_scale_factor=view_get_scale_factor,
        view_set_scale_factor=view_set_scale_factor,

        denoise_to_step=denoise_to_step,

        downsample_factor_before_vae_decode=downsample_factor_before_vae_decode,
        latents=None,
        num_inference_steps=run_args.num_inference_steps,
        num_videos_per_prompt=1,
        generator_seed=run_args.seed,

        output_type = "latent"
    )

    if save_latents:
        torch.save(sphere_SW_latent, os.path.join(output_dir, "sphere_SW_latent.pt"))
        # torch.save(basic_SW_video_frames, os.path.join(output_dir, "basic_SW_video_frames.pt"))

    print("==== Normal Plane Shift Windows Sample ====")

    basic_SW_video_frames, basic_SW_latent = pipeline.basic_sample_shift_multi_windows(
        prompt=prompts,
        img_cond_path=img_cond_path,
        height=run_args.height,
        width=run_args.width,
        frames=16, # run_args.num_inference_steps,
        fps=run_args.fps,
        guidance_scale=run_args.unconditional_guidance_scale,

        init_panorama_latent=sphere_SW_latent,
        use_skip_time=True,
        skip_time_step_idx=denoise_to_step,
        progressive_skip=False,
        total_h=int(equirect_height // downsample_factor_before_vae_decode),
        total_w=int(equirect_width // downsample_factor_before_vae_decode),
        num_windows_h=num_windows_h_2,
        num_windows_w=num_windows_w_2,
        num_windows_f=num_windows_f,
        loop_step=loop_step,
        pano_image_path=pano_image_path,

        latents=None,
        num_inference_steps=run_args.num_inference_steps,
        num_videos_per_prompt=1,
        generator_seed=run_args.seed,
    )

    if save_latents:
        torch.save(basic_SW_latent, os.path.join(output_dir, "basic_SW_latent.pt"))
        torch.save(basic_SW_video_frames, os.path.join(output_dir, "basic_SW_video_frames.pt"))

    save_decoded_video_latents(decoded_video_latents=basic_SW_video_frames,
                               output_path=output_dir,
                               output_name="shift_windows",
                               fps=run_args.fps)

    # basic_video_frames, basic_latent = pipeline.basic_sample(
    #     prompt=prompts,
    #     img_cond_path=img_cond_path,
    #     height=run_args.height,
    #     width=run_args.width,
    #     frames=16, # run_args.num_inference_steps,
    #     fps=run_args.fps,
    #     guidance_scale=run_args.unconditional_guidance_scale,
    #
    #     # init_panorama_latent=None,
    #     # use_skip_time=use_skip_time,
    #     # skip_time_step_idx=skip_time_step_idx,
    #     # progressive_skip=progressive_skip,
    #     # num_windows_h=num_windows_h,
    #     # num_windows_w=num_windows_w,
    #     # loop_step=loop_step,
    #     # pano_image_path=pano_image_path,
    #
    #     latents=None,
    #     num_inference_steps=run_args.num_inference_steps,
    #     num_videos_per_prompt=1,
    #     generator_seed=run_args.seed,
    # )
    #
    # print("==== Basic Sample ====")
    # save_decoded_video_latents(decoded_video_latents=basic_video_frames,
    #                            output_path=output_dir,
    #                            output_name="basic",
    #                            fps=run_args.fps)

    custom_log.reset_log_to_terminal()


if __name__ == "__main__":

    run_args = RunArgs()
    run_args.seed = vargs.seed
    run_args.base_ckpt_path="/home/ljx/360_panorama/FIFO-Diffusion_public/videocrafter_models/i2v_512_v1/model.ckpt"
    run_args.num_inference_steps = run_args.num_partitions * run_args.frame_length

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
    prompt = "azure sky meets turquoise ocean at perfect horizon line, majestic cumulus clouds reflecting in calm waters, golden sunlight highlighting cloud edges, crystal clear waters transitioning from deep blue to turquoise as they approach shore, waves rolling onto the shore with golden sand."
    # "A vibrant cityscape at dusk with fireworks bursting in the purple twilight sky, while cars traffics move along the illuminated roads below"

    # "deep blue sky with clouds moving and galaxy and stars, Ancient Chinese street at night with traditional architecture, golden lantern with shops and houses, ornate wooden buildings with red pillars and curved roofs, few people walking on stone paved streets, glowing shop windows, paper lanterns casting warm light"
    # "green hills with winding country roads, small rural village with stone houses nestled in valley, scattered trees and hedgerows, mixed grassland and farmland, blue sky with dramatic clouds above in bright summer day"

    # "A vibrant cityscape at sunset with fireworks bursting in the sky, while cars traffics move along the illuminated roads below"
    # "A sunny landscape with rolling green hills and dirt paths, fluffy white clouds moving in bright blue sky "
    # "A rocket launch and fly up into the sky at dusk, surrounded by a massive plume of fiery smoke and steam"
    # "green northern lights swirling and shifting across the night sky, reflected in the calm lake waters of a snow-covered landscape"
    # "fireworks burst in the sky above city skyline at night"


    image_folder_name = None

    image_path = None
    image_folder = None

    pano_image_path = "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_beach_2.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/I2V_plane_pano_cases/Genshin_LiYue_4K_Pano.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano360_real_blue_sky_green_grass.jpg"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_volcano.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_fireworkcity_2048-1024.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_greenland_1024-512.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_fireworkcity_1024-512.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/input_img/rocket_launch.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/input_img/lake_under_northern_lights.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/input_img/blue_sky_green_hill_640x1024.png"
    # "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/input_img/city_fireworks_640x1024.png"
    #

    loop_step = 16          # 应大于1, 越小 window 滑动越快, 同时应该保证能整除 width / height / frame, 应该为2的n次方
    num_windows_h = 1
    num_windows_w = 1
    num_windows_f = 1

    # for Sphere Pano Denoising Phrase

    skip_time_step_idx = vargs.skip_time_step
    if skip_time_step_idx >= 0:
        use_skip_time = True
        progressive_skip = True
    else:
        use_skip_time = False
        skip_time_step_idx = 0
        progressive_skip = False    # 从第 0 帧开始逐渐减增大噪声, 直到第 skip_time_step_idx 帧完全是噪声, 按照不 skip 的 time step 来去噪

    denoise_to_step = vargs.denoise_to_step        # 用 Sphere Pano Denoising 去噪 denoise_to_step 步, 然后用一般 SW 去噪


    # Sphere Pano Basic
    downsample_factor_before_vae_decode = 4
    equirect_width = int(1024 * downsample_factor_before_vae_decode)
    equirect_height = int(512 * downsample_factor_before_vae_decode)

    # Notes: 稳定性 / 动态度受到 equirect size 影响的原因可能是 no interp 的稀疏映射 set 加上 paste on image ??

    view_fov = 120 #

    phi_0_first = False

    phi_num = 6

    phi_theta_dict = {
        90:  [0], #[0, 90, 180, 270], # [0, 60, 120, 180, 240, 300], #
        -90: [0], #[0, 90, 180, 270], # [0, 60, 120, 180, 240, 300], #

        75:     [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
        -75:    [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
        60:     [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
        -60:    [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
        45:     [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
        -45:    [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
        0:      [360*t//phi_num for t in range(phi_num)], # [-120, 0, 120], # [0, 45, 90, 135, 180, 225, 270, 315], #
    }

    if phi_0_first:
        phi_theta_dict = OrderedDict(reversed(list(phi_theta_dict.items())))
    phi_prompt_dict = {
        90: "blue sky with white clouds moving",
        75: "blue sky with white clouds moving",
        60: "blue sky with white clouds moving",
        45: "blue sky with white clouds moving",

        0:  "azure sky meets turquoise ocean at perfect horizon line, majestic cumulus clouds reflecting in calm waters, golden sunlight highlighting cloud edges, crystal clear waters transitioning from deep blue to turquoise as they approach shore, waves rolling onto the shore with golden sand.",

        -45: "Clear ocean water with gentle waves rolling onto golden beach",
        -60: "Clear ocean water with gentle waves rolling onto golden beach",
        -75: "Clear ocean water with gentle waves rolling onto golden beach",
        -90: "Clear ocean water with gentle waves rolling onto golden beach",
    }
    # phi_prompt_dict = {
    #     90: "Deep blue night sky with clouds moving, stars twinkling with Milky Way galaxy in the sky ",
    #     75: "deep blue sky with clouds moving above ornate traditional chinese architecture with warm lantern light and mountains",
    #     60: "deep blue sky with clouds moving above ornate traditional chinese architecture with warm lantern light and mountains",
    #     45: "white clouds moving in blue sky, intense sunlight breaking through clouds creating light rays",
    #
    #     0: "Ancient Chinese street at night with traditional architecture, golden lantern with shops and houses, ornate wooden buildings with red pillars and curved roofs, few people walking on stone paved streets, glowing shop windows, paper lanterns casting warm light",
    #
    #     -45: "Ancient Chinese street at night with traditional architecture, golden lantern with shops and houses, ornate wooden buildings with red pillars and curved roofs, few people walking on stone paved streets, glowing shop windows, paper lanterns casting warm light",
    #     -60: "Ancient Chinese street with golden yellow carpet with traditional Chinese patterns on wooden floor and stone pavement",
    #     -75: "Intricate golden yellow carpet with traditional Chinese patterns on wooden floor and stone pavement ",
    #     -90: "Intricate golden yellow carpet with traditional Chinese patterns",
    # }
    # phi_prompt_dict = None

    paste_on_static = True      # 每个 window 去噪得到的 latent 写回到由 pano image 重复并加噪的到的 repeat pano latent 中
                                # 可能用于缓解上下边界的错乱以及 0 问题

    # Sphere Pano SW
    loop_step_theta = vargs.loop_step_theta

    merge_renoised_overlap_latent_ratio = 1

    view_get_scale_factor = 1
    view_set_scale_factor = 1

    # Normal Plane SW
    downsample_factor_before_vae_decode = 1
    num_windows_h_2 = 8
    num_windows_w_2 = 8

    PROJECT_FOLDER = "I2V_SP_4K"
    PROJECT_NOTE = f"I2V_SP_SW-eqW-{equirect_width}_skip-{skip_time_step_idx}_loop-theta-{loop_step_theta}_Hy-{denoise_to_step}"
    PROJECT_NAME = f"{PROJECT_NOTE}" \
                   f"_fov-{view_fov}" \
                   f"{'_phi-0first' if phi_0_first else '_phi-90first'}" \
                   f"_seed-{run_args.seed}" \
                   f"_view_set_scale_factor-{view_set_scale_factor}" \
                   f"_merge_ratio-{merge_renoised_overlap_latent_ratio}" \
                   f"_skip{skip_time_step_idx if use_skip_time else 'NONE'}" \
                   f"{'progressive' if progressive_skip else ''}" \
                   f"_fps-{run_args.fps}" \
                   f"_loop-theta-{loop_step_theta}" \
                   f"{'paste_on_static' if paste_on_static else ''}" \
                   # f"_win{num_windows_f}x{num_windows_h}x{num_windows_w}"


                   # f"_{image_folder_name}_skip-{skip_num_partition}" \
                   # f"{'_dummy-'+str(dummy_prefix_frame) if dummy_prefix_frame>0 else ''}" \
                   # f"{'_prefix-'+str(prefix_sampling_step) if prefix_sampling_step>0 else ''}" \
                   # f"{'_vImgCond' if use_moving_img_cond else ''}" \
                   # f"{'-shift' if (shift_moving_img_cond_by_rank and use_moving_img_cond) else ''}" \
                   # f"_fps-{run_args.fps}" \
                   # f"{'_LA' if run_args.lookahead_denoising else '_no_LA'}"

    save_latents = True

    main(run_args, prompt, image_path, image_folder,
         dummy_prefix_frame=dummy_prefix_frame,
         prefix_sampling_step=prefix_sampling_step,
         skip_num_partition=skip_num_partition,
         use_moving_img_cond=use_moving_img_cond,
         shift_moving_img_cond_by_rank=shift_moving_img_cond_by_rank,
         project_name=PROJECT_NAME,
         project_folder=PROJECT_FOLDER,

         pano_image_path=pano_image_path,
         loop_step=loop_step,
         num_windows_h=num_windows_h,
         num_windows_w=num_windows_w,
         num_windows_f=num_windows_f,

         use_skip_time=use_skip_time,
         skip_time_step_idx=skip_time_step_idx,
         progressive_skip=progressive_skip,

         equirect_width=equirect_width,
         equirect_height=equirect_height,
         phi_theta_dict=phi_theta_dict,
         phi_prompt_dict=phi_prompt_dict,
         # phi_prompt_dict: dict = None,
         view_fov=view_fov,
         loop_step_theta=loop_step_theta,
         merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,

         paste_on_static=paste_on_static,

         view_get_scale_factor=view_get_scale_factor,
         view_set_scale_factor=view_set_scale_factor,

         downsample_factor_before_vae_decode=downsample_factor_before_vae_decode,

         denoise_to_step=denoise_to_step,

         num_windows_h_2=num_windows_h_2,
         num_windows_w_2=num_windows_w_2,

         save_latents=save_latents)
