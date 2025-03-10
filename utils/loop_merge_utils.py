import os

import torch
import imageio
import numpy as np
from PIL import Image



def loop_merge_frames(fifo_loop_sampled_latent, step_size, total_height, total_width, loop_partition):
    """
    step_size should in latent size (divide vae_scale)
    """
    batch_size, num_channel, sampled_length, frame_height, frame_width = fifo_loop_sampled_latent.shape
    assert total_height == frame_height, "[loop_merge_frames] total_height should = frame_height, dont forget to divide vae_scale"
    num_loop_frame = frame_width / step_size / loop_partition
    panorama_loop_tensor = torch.zeros([batch_size, num_channel, int(num_loop_frame)+1, frame_height, total_width]).to(device=fifo_loop_sampled_latent.device)
    # panorama_loop_tensor = torch.zeros([batch_size, num_channel, num_loop_frame, frame_height, total_width]).to(device=fifo_loop_sampled_latent.device)

    for frame_id in range(sampled_length):

        # 分配此 frame 的各个部分应该填充的地方
        curr_top = 0
        curr_down = curr_top + total_height
        curr_left = frame_id * step_size # * loop_partition
        curr_right = curr_left + frame_width

        curr_pano_frame_id = round(frame_id % num_loop_frame)

        curr_frame_tensor = fifo_loop_sampled_latent[:, :, [frame_id], :, :]

        panorama_loop_tensor[:, :, [curr_pano_frame_id], curr_top:curr_down, curr_left:curr_right] = curr_frame_tensor.clone()

        print(f"fifo_loop_sampled_latent[:, :, [{frame_id}], :, :] => [:, :, [{curr_pano_frame_id}], {curr_top}:{curr_down}, {curr_left}:{curr_right}]")

    return panorama_loop_tensor


def tensor2image(batch_tensors):
    img_tensor = torch.squeeze(batch_tensors)  # c,h,w

    image = img_tensor.detach().cpu()
    image = torch.clamp(image.float(), -1., 1.)

    image = (image + 1.0) / 2.0
    image = (image * 255).to(torch.uint8).permute(1, 2, 0)  # h,w,c
    image = image.numpy()
    image = Image.fromarray(image)

    return image


def save_decoded_video_latents(decoded_video_latents, output_path, output_name, fps, save_mp4=True, save_gif=True):

    video_frames_img_list = []

    for frame_idx in range(decoded_video_latents.shape[2]):
        frame_tensor = decoded_video_latents[:, :, [frame_idx]]
        image = tensor2image(frame_tensor)
        video_frames_img_list.append(image)

    print(f"converted {len(video_frames_img_list)} frame tensors")

    if save_mp4:
        mp4_save_path = os.path.join(output_path, f"{output_name}.mp4")
        imageio.mimsave(mp4_save_path, video_frames_img_list, fps=fps)
        print(f"pano video saved to -> {mp4_save_path}")

    # if save_gif:
    #     gif_save_path = os.path.join(output_path, f"{output_name}.gif")
    #     imageio.mimsave(gif_save_path, video_frames_img_list, duration=int(1000 / fps), loop=0)
    #     print(f"pano gif saved to -> {gif_save_path}")


if __name__ == "__main__":

    loop_sampled_tensor_path = "/home/ljx/_temp/fifo_free_traj_turbo_test/results/videocraft_v2_fifo/random_noise/1007_23-38-57-_city_fireworks_1638_cir-1024_skip-1_prefix-36_fps-8_no_LA/fifo_loop_sampled_latent.pt"
    total_video_length = 1024
    fps = 8
    output_path = "/home/ljx/_temp/fifo_free_traj_turbo_test/results/videocraft_v2_fifo/random_noise/1007_23-38-57-_city_fireworks_1638_cir-1024_skip-1_prefix-36_fps-8_no_LA"

    loop_partition = 0.5 # (1024+512) // 512  # 1 则等于不loop, 纯从左划到右, 值越大loop越多次, 理论上应该设置为 total_width/frame_width

    loop_sampled_tensor = torch.load(loop_sampled_tensor_path)
    print(f"Loaded {loop_sampled_tensor.shape}")

    loop_sampled_tensor = loop_sampled_tensor[:, :, -total_video_length:]

    panorama_loop_tensor = loop_merge_frames(fifo_loop_sampled_latent=loop_sampled_tensor,
                                             step_size=1, total_height=320, total_width=1024+512, loop_partition=loop_partition)
    # panorama_loop_tensor = panorama_loop_tensor.to(device="cpu")

    panorama_video_frames = []

    for frame_idx in range(panorama_loop_tensor.shape[2]):
        frame_tensor = panorama_loop_tensor[:, :, [frame_idx]]
        image = tensor2image(frame_tensor)
        # if save_frames:
        #     fifo_path = os.path.join(fifo_dir, f"{i}.png")
        #     image.save(fifo_path)
        panorama_video_frames.append(image)

    print(f"converted {len(panorama_video_frames)} frame tensors")

    panorama_save_path = os.path.join(output_path, f"merged_pano-{loop_partition}.mp4")
    imageio.mimsave(panorama_save_path, panorama_video_frames, fps=fps)
    print(f"pano video saved to -> {panorama_save_path}")

    panorama_save_path = os.path.join(output_path, f"merged_pano-{loop_partition}.gif")
    imageio.mimsave(panorama_save_path, panorama_video_frames, duration=int(1000 / fps))
    print(f"pano gif saved to -> {panorama_save_path}")

    # loop_sampled_tensor_path = "/home/ljx/_temp/fifo_free_traj_turbo_test/results/videocraft_v2_fifo/random_noise/1007_13-32-57-test_no_LA_and_fix_pipe_city_fireworks_1638-128_skip-1_prefix-36_fps-8_no_LA/fifo_loop_sampled_latent.pt"
    # total_video_length = 128
    # fps = 8
    # output_path = "/home/ljx/_temp/fifo_free_traj_turbo_test/results/videocraft_v2_fifo/random_noise/1007_13-32-57-test_no_LA_and_fix_pipe_city_fireworks_1638-128_skip-1_prefix-36_fps-8_no_LA"
    #
    # loop_partition = 2 # (1024+512) // 512  # 1 则等于不loop, 纯从左划到右, 值越大loop越多次, 理论上应该设置为 total_width/frame_width
    #
    # loop_sampled_tensor = torch.load(loop_sampled_tensor_path)
    # print(f"Loaded {loop_sampled_tensor.shape}")
    #
    # loop_sampled_tensor = loop_sampled_tensor[:, :, -total_video_length:]
    #
    # panorama_loop_tensor = loop_merge_frames(fifo_loop_sampled_latent=loop_sampled_tensor,
    #                                          step_size=4, total_height=320, total_width=1024, loop_partition=loop_partition)
    # # panorama_loop_tensor = panorama_loop_tensor.to(device="cpu")
    #
    # panorama_video_frames = []
    #
    # for frame_idx in range(panorama_loop_tensor.shape[2]):
    #     frame_tensor = panorama_loop_tensor[:, :, [frame_idx]]
    #     image = tensor2image(frame_tensor)
    #     # if save_frames:
    #     #     fifo_path = os.path.join(fifo_dir, f"{i}.png")
    #     #     image.save(fifo_path)
    #     panorama_video_frames.append(image)
    #
    # panorama_save_path = os.path.join(output_path, f"merged_pano-{loop_partition}.mp4")
    # print(f"pano video saved to -> {panorama_save_path}")
    # imageio.mimsave(panorama_save_path, panorama_video_frames, fps=fps)




