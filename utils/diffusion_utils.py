import torch
import torch.nn.functional as F

def get_w_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    Args:
    timesteps: torch.Tensor: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings
    Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def padding_latents_at_front(source_latents, front_padding_num):
    latents_list = []
    for i in range(front_padding_num):
        latents_list.append(source_latents[:, :, [0]])
    latents_list.append(source_latents)
    padded_latents = torch.cat(latents_list, dim=2)
    return padded_latents


def padding_latents_at_end(source_latents, end_padding_num):
    latents_list = [source_latents]
    for i in range(end_padding_num):
        latents_list.append(source_latents[:, :, [-1]])
    padded_latents = torch.cat(latents_list, dim=2)
    return padded_latents


def resize_video_latent(input_latent, target_height, target_width, mode='bilinear', align_corners=False):
    """
    对形状为 (batch, channel, frame, h, w) 的张量在 H 和 W 维度进行插值放大。

    参数:
    - video_latent_0: 输入张量，形状为 (batch, channel, frame, h, w)
    - target_h: 目标高度
    - target_w: 目标宽度
    - mode: 插值方式，如 'bilinear', 'nearest' 等
    - align_corners: 插值时是否对齐角点（仅在某些插值模式下有效）

    返回:
    - upsampled: 插值后的张量，形状为 (batch, channel, frame, target_h, target_w)
    """
    if mode == 'nearest':
        align_corners = None

    batch, channel, frame, h, w = input_latent.shape

    # 形状变为 (batch, frame, channel, h, w)
    input_latent = input_latent.permute(0, 2, 1, 3, 4)

    # 将 (batch, frame, channel, h, w) 重塑为 (batch * frame, channel, h, w)
    video_reshaped = input_latent.view(batch * frame, channel, h, w)

    # 执行 2D 插值
    upsampled = F.interpolate(video_reshaped, size=(target_height, target_width), mode=mode, align_corners=align_corners)

    # 将插值后的张量重塑回 (batch, frame, channel, target_h, target_w)
    upsampled = upsampled.view(batch, frame, channel, target_height, target_width)

    # 调整维度顺序为 (batch, channel, frame, target_h, target_w)
    upsampled = upsampled.permute(0, 2, 1, 3, 4)

    return upsampled

    # # TODO : 待修改, 不支持 bilinear, 考虑使用新版的实现
    # """
    # 缩放输入张量的高度和宽度到指定大小。
    #
    # 参数:
    #     x (torch.Tensor): 输入张量，形状为 (batch, channel, frame, h, w)。
    #     target_size (tuple): 目标大小 (new_h, new_w)。
    #     mode (str): 插值模式，默认为 'bilinear'。支持 'nearest', 'bilinear', 'bicubic', 'trilinear' 等。
    #     align_corners (bool): 是否在插值时对齐角点，默认为 False。
    #
    # 返回:
    #     torch.Tensor: 缩放后的张量，形状为 (batch, channel, frame, new_h, new_w)。
    # """
    # if mode == 'nearest':
    #     align_corners = None
    # resized_latent = F.interpolate(input_latent, size=(input_latent.size(2), target_height, target_width), mode=mode, align_corners=align_corners)
    #
    # return resized_latent



    # # 1. 将维度 permute，使得 frame 维度在 batch 维度之后
    # input_latent = input_latent.permute(0, 2, 1, 3, 4)  # 形状变为 (batch, frame, channel, h, w)
    #
    # # 2. 重塑张量，将 batch 和 frame 维度合并
    # batch_size, frame_size, channels, height, width = input_latent.shape
    # input_latent = input_latent.reshape(batch_size * frame_size, channels, height, width)  # 新形状为 (batch * frame, channel, h, w)
    #
    # # 3. 使用 interpolate 函数缩放高度和宽度
    # if mode == 'nearest':
    #     align_corners = None
    # input_latent = F.interpolate(input_latent, size=target_size, mode=mode, align_corners=align_corners)  # 缩放后的形状为 (batch * frame, channel, new_h, new_w)
    #
    # # 4. 重塑回原来的维度
    # input_latent = input_latent.reshape(batch_size, frame_size, channels, *target_size)  # 形状为 (batch, frame, channel, new_h, new_w)
    #
    # # 5. 将维度 permute 回原来的顺序
    # input_latent = input_latent.permute(0, 2, 1, 3, 4)  # 最终形状为 (batch, channel, frame, new_h, new_w)
    #
    # return input_latent


def upsample_video_latent(video_latent_0, target_h, target_w, mode='bilinear', align_corners=False):
    """
    对形状为 (batch, channel, frame, h, w) 的张量在 H 和 W 维度进行插值放大。

    参数:
    - video_latent_0: 输入张量，形状为 (batch, channel, frame, h, w)
    - target_h: 目标高度
    - target_w: 目标宽度
    - mode: 插值方式，如 'bilinear', 'nearest' 等
    - align_corners: 插值时是否对齐角点（仅在某些插值模式下有效）

    返回:
    - upsampled: 插值后的张量，形状为 (batch, channel, frame, target_h, target_w)
    """
    batch, channel, frame, h, w = video_latent_0.shape

    # 将 (batch, channel, frame, h, w) 重塑为 (batch * frame, channel, h, w)
    video_reshaped = video_latent_0.view(batch * frame, channel, h, w)

    # 执行 2D 插值
    upsampled = F.interpolate(video_reshaped, size=(target_h, target_w), mode=mode, align_corners=align_corners)

    # 将插值后的张量重塑回 (batch, frame, channel, target_h, target_w)
    upsampled = upsampled.view(batch, frame, channel, target_h, target_w)

    # 调整维度顺序为 (batch, channel, frame, target_h, target_w)
    upsampled = upsampled.permute(0, 2, 1, 3, 4)

    return upsampled

def expand_per_frame_prompts(origin_prompt_dict, total_frames_num):
    # 先将原始prompt字典的键按升序排序
    sorted_keys = sorted(origin_prompt_dict.keys())
    frame_prompts = {}

    # 遍历所有帧
    for i in range(total_frames_num):
        # 找到当前帧对应的 prompt
        for start_frame in sorted_keys[::-1]:  # 从后向前找，以便找到最后一个小于等于当前帧的起始帧
            if i >= start_frame:
                frame_prompts[i] = origin_prompt_dict[start_frame]
                break

    return frame_prompts


