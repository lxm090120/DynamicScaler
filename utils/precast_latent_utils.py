import os
import re

import torch
from torchvision import transforms
from PIL import Image

from pipeline.t2v_turbo_scheduler import T2VTurboScheduler
from utils.utils_freetraj import get_freq_filter, freq_mix_3d, plan_path

import torch.nn.functional as F


def _extract_number(filename):
    match = re.match(r'window_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'window_(\d+)\.jpg$', filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # 如果没有匹配到，返回无穷大以便排到最后

def _load_and_preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),                      # 调整图片大小
        transforms.ToTensor(),                              # 将图片转换为Tensor
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # 将[0,1]标准化
        transforms.Lambda(lambda x: x * 2.0 - 1.0)          # 转换到[-1,1]
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)  # 增加batch维度


def encode_image_to_latent_tensor(pretrained_t2v, image_path, image_size):
    img_tensor = _load_and_preprocess_image(image_path, image_size)
    img_latent = pretrained_t2v.encode_first_stage_2DAE(img_tensor.unsqueeze(1).unsqueeze(0).to(dtype=pretrained_t2v.dtype,
                                                                                                device=pretrained_t2v.device))
    # [ c, h, w ] -> [ b, c, t, h, w ]
    return img_latent


def get_img_list_from_folder(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    image_files_name_list = sorted(image_files, key=_extract_number)
    image_path_list = [os.path.join(image_folder, image_name) for image_name in image_files_name_list]
    return image_path_list

def encode_images_list_to_latent_tensor(pretrained_t2v, image_folder, image_size, image_path_list=None):

    if image_path_list is None:
        image_path_list = get_img_list_from_folder(image_folder)

    latent_list = []

    for image_path in image_path_list:
        img_tensor = _load_and_preprocess_image(image_path, image_size)
        img_latent = pretrained_t2v.encode_first_stage_2DAE(img_tensor.unsqueeze(1).unsqueeze(0).to(dtype=pretrained_t2v.dtype, device=pretrained_t2v.device))
        latent_list.append(img_latent)
        # [ c, h, w ] -> [ b, c, t, h, w ]

    video_latent = torch.cat(latent_list, dim=2)

    return video_latent



def _add_noise_to_x0_tensors(scheduler: T2VTurboScheduler, target_time_step: int, x0_tensors: torch.FloatTensor):

    assert len(x0_tensors.shape) == 5, 'x0_tensors shape ERR'

    noise = torch.randn(x0_tensors.shape).to(dtype=x0_tensors.dtype, device=x0_tensors.device)
    time_step_tensors = torch.full_like(x0_tensors, target_time_step).to(device=x0_tensors.device)
    x_t_tensors = scheduler.add_noise(original_samples=x0_tensors,
                                      noise=noise,
                                      timesteps=target_time_step)
    return x_t_tensors


def _prepare_precast_latents_free_traj(
        num_channels_latents,
        frames,
        height,
        width,
        device,
        background_tensors,
        foreground_tensors,
        planed_full_path,
        use_filter=True,
):
    total_shape = (
        1,
        1,  # Notes : 强制 batch size 和 n_samples 都为 1, (可能会有问题?)
        num_channels_latents,
        frames,
        height,
        width,
    )
    print('noise_flow...', 'total_shape', total_shape)

    assert background_tensors.shape == total_shape , "Back Ground Tensor Shape Not Match"

    x_T_total = background_tensors

    BOX_SIZE_H = planed_full_path[0][1] - planed_full_path[0][0]
    BOX_SIZE_W = planed_full_path[0][3] - planed_full_path[0][2]
    PATHS = planed_full_path
    sub_h = int(BOX_SIZE_H * height)
    sub_w = int(BOX_SIZE_W * width)

    for i in range(frames):
        h_start = int(PATHS[i][0] * height)
        h_end = h_start + sub_h
        w_start = int(PATHS[i][2] * width)
        w_end = w_start + sub_w

        scaled_fg_tensor = F.interpolate(foreground_tensors[0, :, :, i], size=(sub_h, sub_w), mode='bilinear', align_corners=False).unsqueeze(0)

        # no mix
        x_T_total[:, :, :, i, h_start:h_end, w_start:w_end] = scaled_fg_tensor

    if use_filter:

        filter_shape = [
            1,
            num_channels_latents,
            frames,
            height,
            width
        ]

        freq_filter = get_freq_filter(
            filter_shape,
            device=device,
            filter_type='butterworth',
            n=4,
            d_s=0.25,
            d_t=0.1
        )

        x_T_rand = torch.randn([1, 1, num_channels_latents, frames, height, width], device=device)
        x_T_total = freq_mix_3d(x_T_total.to(dtype=torch.float32), x_T_rand, LPF=freq_filter)

    latents = x_T_total[0]  # Notes: 消除了 n_sample 维度 (1)

    # scale the initial noise by the standard deviation required by the scheduler
    # latents = latents * self.scheduler.init_noise_sigma
    return latents


def expand_tensor_node_to_tail(source_tensor, target_len):
    # 取出首元素和尾元素

    first = source_tensor[:, :, [0]]
    middle = source_tensor[:, :, 1:-1]
    last = source_tensor[:, :, [-1]]

    # 生成中间部分的正序和倒序
    middle_tensor = torch.tensor(middle)
    reversed_middle = middle_tensor.flip(dims=[2])

    # 初始化结果列表，并设置当前元素为首元素
    result = [first]
    use_reversed = False  # 控制交替使用正序或倒序

    # 循环添加元素直到达到长度k
    while len(result) < target_len:
        if use_reversed:
            # 添加倒序中间部分和尾元素
            result.append(reversed_middle.clone())
            if len(result) < target_len:  # 检查是否还需要添加元素
                result.append(first)
        else:
            # 添加正序中间部分和尾元素
            result.append(middle_tensor.clone())
            if len(result) < target_len:  # 检查是否还需要添加元素
                result.append(last)

        # 切换方向
        use_reversed = not use_reversed

    # 截取结果到指定长度
    return torch.cat(result, dim=2)[:, :, :target_len]


def prepare_skip_latents_free_traj(
        scheduler: T2VTurboScheduler,
        target_time_step: int,  # Notes: 目标噪声强度 (0~1000), 注意需要先 scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
        num_channels_latents,
        frames,
        height,
        width,
        device,
        background_tensors_x0,
        foreground_tensors_x0,
        planed_full_path,
        just_collage=False,
        use_filter=True,
):
    if background_tensors_x0.shape[2] == 1:
        extended_background_tensors = torch.cat([background_tensors_x0] * frames, dim=2)
    else:
        extended_background_tensors = expand_tensor_node_to_tail(source_tensor=background_tensors_x0,
                                                                 target_len=frames)

    extended_foreground_tensors = expand_tensor_node_to_tail(source_tensor=foreground_tensors_x0,
                                                                target_len=frames)
    if not just_collage:
        # TODO: CHECK
        extended_background_tensors = _add_noise_to_x0_tensors(scheduler=scheduler,
                                                             target_time_step=target_time_step,
                                                             x0_tensors=extended_background_tensors)
        extended_foreground_tensors = _add_noise_to_x0_tensors(scheduler=scheduler,
                                                             target_time_step=target_time_step,
                                                             x0_tensors=extended_foreground_tensors)

    precast_latents = _prepare_precast_latents_free_traj(
        num_channels_latents=num_channels_latents,
        frames=frames,
        height=height,
        width=width,
        device=device,
        background_tensors=extended_background_tensors.unsqueeze(0),
        foreground_tensors=extended_foreground_tensors.unsqueeze(0),
        planed_full_path=planed_full_path,
        use_filter=use_filter,
    )

    return precast_latents


