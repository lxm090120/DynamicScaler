import torch
import torch.nn.functional as F


class PanoramaTensor:
    def __init__(self, equirect_tensor):
        """
        初始化 PanoramaTensor。

        参数：
            equirect_tensor (torch.Tensor): 任意形状的等距投影全景张量，最后两个维度必须是 H 和 W，
                                           且宽度必须是高度的两倍。
                                           例如：
                                               - [H, W] (暂不支持, 需要手动管理为 [1, H, W])
                                               - [C, H, W]
                                               - [B, C, H, W]
                                               - [B, N, C, H, W]
                                               - 以及更高维度，只要最后两个维度是 H 和 W。
        """
        assert equirect_tensor.dim() >= 2, "输入张量必须至少具有两个维度 [H, W]"
        H, W = equirect_tensor.shape[-2], equirect_tensor.shape[-1]
        assert W == 2 * H, "宽度必须是高度的两倍"

        # 如果输入张量是 [H, W]，自动增加一个通道维度
        if equirect_tensor.dim() == 2:
            equirect_tensor = equirect_tensor.unsqueeze(0)  # [1, H, W]

        # 获取通道数
        C = equirect_tensor.shape[-3] if equirect_tensor.dim() >= 3 else 1
        if equirect_tensor.dim() == 3:
            C = equirect_tensor.shape[0]
        elif equirect_tensor.dim() > 3:
            C = equirect_tensor.shape[-3]

        self.equirect_tensor = equirect_tensor.clone()
        self.C = C
        self.H = H
        self.W = W
        self.device = equirect_tensor.device
        self.dtype = equirect_tensor.dtype

    # @staticmethod
    # def init_equirect_tensor(image_path, device='cpu'):
    #     """
    #     从图像文件初始化等距投影全景张量。
    #
    #     参数：
    #         image_path (str): 全景图像的文件路径。
    #         device (str or torch.device): 使用的设备（'cpu' 或 'cuda'）。
    #
    #     返回：
    #         PanoramaTensor: 初始化后的 PanoramaTensor 实例。
    #     """
    #     # 读取全景图像并转换为 PyTorch 张量
    #     equirect_img = Image.open(image_path).convert('RGB')
    #     equirect_tensor = TF.to_tensor(equirect_img).to(device)
    #     return PanoramaTensor(equirect_tensor)

    def get_view_tensor_interpolate(self, fov, theta, phi, width, height,
                                    interpolate_mode='bilinear', interpolate_align_corners=True):
        """
        从全景张量中提取指定视角的视图张量。

        参数：
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
            width (int): 视图的宽度。
            height (int): 视图的高度。

        返回：
            torch.Tensor: 视图张量，形状为 [*, C, height, width]，其中 * 表示任意数量的前导维度。
        """
        # 获取前导维度
        leading_dims = self.equirect_tensor.shape[:-3] if self.equirect_tensor.dim() > 3 else ()
        B = torch.prod(torch.tensor(leading_dims)) if len(leading_dims) > 0 else 1
        B = int(B)

        # Reshape to [B, C, H, W]
        pano = self.equirect_tensor.view(-1, self.C, self.H, self.W)  # [B, C, H, W]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        # Normalize grid to [-1, 1] for grid_sample
        grid_u = (u / (self.W - 1)) * 2 - 1  # [height, width]
        grid_v = (v / (self.H - 1)) * 2 - 1  # [height, width]
        grid = torch.stack((grid_u, grid_v), dim=-1)  # [height, width, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, height, width, 2]

        # 使用 grid_sample 进行采样
        view = F.grid_sample(pano, grid, mode=interpolate_mode, padding_mode='border',
                             align_corners=interpolate_align_corners)  # [B, C, height, width]

        # Reshape back to original leading dimensions
        if len(leading_dims) > 0:
            view = view.view(*leading_dims, self.C, height, width)  # [*, C, height, width]
        else:
            view = view.squeeze(0)  # [C, height, width]

        return view

    def get_view_tensor_no_interpolate(self, fov, theta, phi, width, height):
        """
        从全景张量中提取指定视角的视图张量。

        参数：
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
            width (int): 视图的宽度。
            height (int): 视图的高度。

        返回：
            torch.Tensor: 视图张量，形状为 [*, C, height, width]，其中 * 表示任意数量的前导维度
            torch.Tensor: 掩码张量，形状为 [height, width]，标记未填充的区域（0 为填充，1 为未填充）
        """
        # 获取前导维度
        leading_dims = self.equirect_tensor.shape[:-3] if self.equirect_tensor.dim() > 3 else ()
        B = torch.prod(torch.tensor(leading_dims)) if len(leading_dims) > 0 else 1
        B = int(B)

        # Reshape to [B, C, H, W]
        pano = self.equirect_tensor.view(-1, self.C, self.H, self.W)  # [B, C, H, W]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        sampled_view, unsampled_mask = self._sample_equirect_tensor_nearest(pano, u, v)

        # Reshape back to original leading dimensions
        if len(leading_dims) > 0:
            sampled_view = sampled_view.view(*leading_dims, self.C, height, width)  # [*, C, height, width]
        else:
            sampled_view = sampled_view.squeeze(0)  # [C, height, width]

        return sampled_view, unsampled_mask

    def set_view_tensor(self, view_tensor, fov, theta, phi):
        """
        使用最近邻插值将编辑后的视图张量写回等距投影全景张量中。

        参数：
            view_tensor (torch.Tensor): 编辑后的视图张量，形状为 [*, C, height, width] 或 [C, height, width]。
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
        """
        # 如果输入张量是 [C, H, W]，自动增加批次维度
        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)  # [1, C, H, W]

        # 获取前导维度
        leading_dims = self.equirect_tensor.shape[:-3]  # [*, C, H, W]
        B = torch.prod(torch.tensor(leading_dims)) if len(leading_dims) > 0 else 1
        B = int(B)

        # Reshape to [B, C, H, W]
        pano = self.equirect_tensor.view(-1, self.C, self.H, self.W)  # [B, C, H, W]
        view = view_tensor.view(-1, self.C, view_tensor.shape[-2], view_tensor.shape[-1]).clone()  # [B, C, height, width]

        # 创建图像平面的坐标系
        width, height = view.shape[-1], view.shape[-2]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        # 最近邻映射
        u_nn = torch.round(u).long().clamp(0, self.W - 1)
        v_nn = torch.round(v).long().clamp(0, self.H - 1)

        # 展开视图张量和全景张量以进行索引
        flat_view = view.view(B, self.C, -1)  # [B, C, height*width]
        flat_pano = pano.view(B, self.C, -1)  # [B, C, H_pano*W_pano]

        # 计算全景张量中的线性索引
        linear_indices = (v_nn * self.W + u_nn).view(B, -1)  # [B, height*width]

        # 使用 scatter_ 进行批量赋值
        flat_pano.scatter_(2, linear_indices.unsqueeze(1).expand(-1, self.C, -1), flat_view)

        # 将全景张量重塑回原始形状
        pano = flat_pano.view(B, self.C, self.H, self.W)
        self.equirect_tensor = pano.view(*leading_dims, self.C, self.H, self.W) if B > 1 else pano.squeeze(0)

    def set_view_tensor_bilinear(self, view_tensor, fov, theta, phi):

        """
        使用双线性插值将编辑后的视图张量写回等距投影全景张量中。

        参数：
            view_tensor (torch.Tensor): 编辑后的视图张量，形状为 [..., C, height, width]。
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
        """
        # 如果输入张量是 [C, H, W]，自动增加批次维度
        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)  # [1, C, H, W]

        # 获取前导维度
        leading_dims = view_tensor.shape[:-3]  # [*, C, H, W]
        B = torch.prod(torch.tensor(leading_dims)).item() if len(leading_dims) > 0 else 1
        view = view_tensor.view(B, self.C, view_tensor.shape[-2], view_tensor.shape[-1]).clone()  # [B, C, height, width]

        # 创建图像平面的坐标系
        width, height = view.shape[-1], view.shape[-2]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        # 获取整数和小数部分
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = (u0 + 1) % self.W
        v1 = torch.clamp(v0 + 1, 0, self.H - 1)

        # 计算双线性插值权重
        du = (u - u0.float()).unsqueeze(0)  # [1, height, width]
        dv = (v - v0.float()).unsqueeze(0)  # [1, height, width]

        w00 = (1 - du) * (1 - dv)  # [1, height, width]
        w01 = (1 - du) * dv
        w10 = du * (1 - dv)
        w11 = du * dv

        # 展开视图张量和权重
        view_flat = view.view(B, self.C, -1)  # [B, C, height*width]
        w00 = w00.view(-1)  # [height*width]
        w01 = w01.view(-1)
        w10 = w10.view(-1)
        w11 = w11.view(-1)

        # 展开索引
        idx00 = (v0 * self.W + u0).view(-1)  # [height*width]
        idx01 = (v1 * self.W + u0).view(-1)
        idx10 = (v0 * self.W + u1).view(-1)
        idx11 = (v1 * self.W + u1).view(-1)

        # 遍历批次和通道维度
        for b in range(B):
            # 累加器和权重和
            accumulator = torch.zeros_like(self.equirect_tensor.view(B, self.C, -1)[b])  # [B, C, H_pano*W_pano]
            weight_sum = torch.zeros_like(self.equirect_tensor.view(B, self.C, -1)[b])  # [B, C, H_pano*W_pano]

            for c in range(self.C):
                # 累加
                accumulator[c].index_add_(0, idx00, view_flat[b, c] * w00)
                accumulator[c].index_add_(0, idx01, view_flat[b, c] * w01)
                accumulator[c].index_add_(0, idx10, view_flat[b, c] * w10)
                accumulator[c].index_add_(0, idx11, view_flat[b, c] * w11)

                weight_sum[c].index_add_(0, idx00, w00)
                weight_sum[c].index_add_(0, idx01, w01)
                weight_sum[c].index_add_(0, idx10, w10)
                weight_sum[c].index_add_(0, idx11, w11)

            # 防止除以零
            mask = weight_sum > 0
            self.equirect_tensor.view(B, self.C, -1)[b][mask] = accumulator[mask] / weight_sum[mask]

    def set_view_tensor_no_interpolation(self, view_tensor, fov, theta, phi):
        """
        使用无插值的方式将编辑后的视图张量写回等距投影全景张量中。
        接受映射可能存在空洞。

        参数：
            view_tensor (torch.Tensor): 编辑后的视图张量，形状为 [*, C, height, width] 或 [C, height, width]。
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
        """
        # 如果输入张量是 [C, H, W]，自动增加批次维度
        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)  # [1, C, H, W]

        # 获取前导维度
        leading_dims = view_tensor.shape[:-3]  # [*, C, H, W]
        B = torch.prod(torch.tensor(leading_dims)).item() if len(leading_dims) > 0 else 1

        # 将视图张量重塑为 [B, C, H, W]
        view = view_tensor.view(B, self.C, view_tensor.shape[-2], view_tensor.shape[-1])  # [B, C, height, width]

        # 创建图像平面的坐标系
        width, height = view.shape[-1], view.shape[-2]

        # 计算 u, v 坐标
        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)  # [height, width]

        # 使用 floor 获取整数坐标（不进行取整）
        u_int = torch.floor(u).long()
        v_int = torch.floor(v).long()

        # 检查有效索引（在全景范围内）
        valid_mask = (u_int >= 0) & (u_int < self.W) & (v_int >= 0) & (v_int < self.H)  # [height, width]

        # 展开视图张量和全景张量以进行索引
        view_flat = view.view(B, self.C, -1)  # [B, C, height*width]
        pano_flat = self.equirect_tensor.view(-1, self.C, self.H * self.W)  # [B, C, H*W]

        # 计算全景张量中的线性索引
        linear_indices = (v_int * self.W + u_int).view(-1)  # [B * height * width]

        # 仅选择有效的索引和对应的视图像素值
        valid_linear_indices = linear_indices[valid_mask.view(-1)]  # [num_valid_pixels]
        valid_view = view_flat.reshape(B * self.C, -1)[:, valid_mask.view(-1)]  # [B*C, num_valid_pixels]

        # 将视图像素值写回全景张量
        pano_flat = pano_flat.reshape(B * self.C, -1)  # [B*C, H*W]
        pano_flat[:, valid_linear_indices] = valid_view

        # 将全景张量重塑回原始形状
        self.equirect_tensor = pano_flat.reshape(self.equirect_tensor.shape)

    def _sample_equirect_tensor_nearest(self, pano, u, v):
        """
        根据给定的 u, v 坐标从等距投影全景张量中采样像素值（不插值）。

        参数：
            pano (torch.Tensor): 重排到 [B, C, height, width] 的 self.equirect_tensor
            u (torch.Tensor): 水平方向的像素坐标，形状为 [height, width]
            v (torch.Tensor): 垂直方向的像素坐标，形状为 [height, width]

        返回：
            torch.Tensor: 视图张量，形状为 [C, height, width]
            torch.Tensor: 掩码张量，形状为 [height, width]，标记未填充的区域（0 为填充，1 为未填充）
        """
        # 计算整数部分的坐标
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()

        # 确保坐标在有效范围内
        u0 = u0 % self.W
        v0 = torch.clamp(v0, 0, self.H - 1)

        # 获取像素值
        sampled_view = pano[:, :, v0, u0].clone()  # [B, C, height, width]

        # 创建掩码张量，默认值为1（未填充区域）
        unsampled_mask = torch.ones_like(u0, dtype=self.dtype, device=self.device)  # [height, width]

        # 标记填充的区域为0
        valid_mask = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        unsampled_mask[~valid_mask] = 0

        # 将不在有效范围内的区域填充为0
        sampled_view[:, :, ~valid_mask] = 0

        return sampled_view, unsampled_mask

    def _get_uv(self, fov, theta, phi, width, height, focal_length=None):

        # 将角度转换为弧度
        fov_rad = torch.deg2rad(torch.tensor(fov, dtype=self.dtype, device=self.device))
        theta_rad = torch.deg2rad(torch.tensor(theta, dtype=self.dtype, device=self.device))
        phi_rad = torch.deg2rad(torch.tensor(phi, dtype=self.dtype, device=self.device))

        # 焦距
        if focal_length is None:
            f = 0.5 * width / torch.tan(fov_rad / 2)
        else:
            f = focal_length

        # 创建图像平面的坐标系
        # width, height = view.shape[-1], view.shape[-2]
        x = torch.linspace(-width / 2, width / 2 - 1, steps=width, dtype=self.dtype, device=self.device)
        y = torch.linspace(-height / 2, height / 2 - 1, steps=height, dtype=self.dtype, device=self.device)
        yv, xv = torch.meshgrid(y, x, indexing='ij')  # [height, width]

        # 计算图像平面上的三维方向向量
        zv = torch.full_like(xv, f)
        xyz = torch.stack([xv, yv, zv], dim=-1)  # [height, width, 3]
        norm = torch.norm(xyz, dim=-1, keepdim=True)
        xyz_norm = xyz / norm  # 单位化

        # 旋转矩阵（先绕 X 轴旋转 phi，再绕 Y 轴旋转 theta）
        R_phi = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(phi_rad), -torch.sin(phi_rad)],
            [0, torch.sin(phi_rad), torch.cos(phi_rad)]
        ], dtype=self.dtype, device=self.device)

        R_theta = torch.tensor([
            [torch.cos(theta_rad), 0, torch.sin(theta_rad)],
            [0, 1, 0],
            [-torch.sin(theta_rad), 0, torch.cos(theta_rad)]
        ], dtype=self.dtype, device=self.device)

        R = torch.matmul(R_theta, R_phi)  # [3, 3]

        # 应用旋转矩阵
        xyz_rot = torch.matmul(xyz_norm.view(-1, 3), R.t()).view(height, width, 3)  # [height, width, 3]

        # 计算球面坐标（经度和纬度）
        lon = torch.atan2(xyz_rot[..., 0], xyz_rot[..., 2])  # [-pi, pi]
        lat = torch.asin(xyz_rot[..., 1])  # [-pi/2, pi/2]

        # 将经度映射到 [0, 2*pi)
        lon = (lon + 2 * torch.pi) % (2 * torch.pi)  # [0, 2*pi)

        # 将球面坐标映射到等距投影全景图像像素坐标
        u = lon / (2 * torch.pi) * (self.W - 1)  # [height, width]
        v = (lat + torch.pi / 2) / torch.pi * (self.H - 1)  # [height, width]

        return u, v



class PanoramaLatentProxy:
    def __init__(self, equirect_tensor):
        """
        初始化 PanoramaLatentProxy。

        参数：
            equirect_tensor (torch.Tensor): 形状为 [B, C, N, H, W] 的等距投影全景张量。
        """
        assert equirect_tensor.dim() >= 4, "输入张量必须至少具有四个维度 [B, C, N, H, W]"
        self.original_shape = equirect_tensor.shape
        B, C, N, H, W = self.original_shape

        # 交换维度以匹配 PanoramaTensor 的要求 [B, N, C, H, W]
        equirect_tensor_reordered = equirect_tensor.permute(0, 2, 1, 3, 4)

        # 创建 PanoramaTensor 实例
        self.panorama_tensor = PanoramaTensor(equirect_tensor_reordered)

    def get_view_tensor_interpolate(self, fov, theta, phi, width, height,
                                    interpolate_mode='bilinear', interpolate_align_corners=True):
        """
        使用 PanoramaTensor 提取视图张量，并将形状转换回 [B, C, N, height, width]。

        参数：
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
            width (int): 视图的宽度。
            height (int): 视图的高度。

        返回：
            torch.Tensor: 视图张量，形状为 [B, C, N, height, width]。
        """
        # 使用 PanoramaTensor 提取视图张量
        view = self.panorama_tensor.get_view_tensor_interpolate(
            fov, theta, phi, width, height, interpolate_mode, interpolate_align_corners)

        # 将形状转换回 [B, C, N, height, width]
        B, N, C, H, W = view.shape
        return view.permute(0, 2, 1, 3, 4).clone()

    def get_view_tensor_no_interpolate(self, fov, theta, phi, width, height):
        """
        使用 PanoramaTensor 提取视图张量（无插值），并将形状转换回 [B, C, N, height, width]。

        参数：
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
            width (int): 视图的宽度。
            height (int): 视图的高度。

        返回：
            tuple: 视图张量和掩码张量，形状分别为 [B, C, N, height, width] 和 [B, N, height, width]。
        """
        # 使用 PanoramaTensor 提取视图张量
        view, mask = self.panorama_tensor.get_view_tensor_no_interpolate(fov, theta, phi, width, height)

        # 将视图张量形状转换回 [B, C, N, height, width]
        B, N, C, H, W = view.shape
        view = view.permute(0, 2, 1, 3, 4)

        return view, mask

    def set_view_tensor(self, view_tensor, fov, theta, phi):
        """
        将编辑后的视图张量写回等距投影全景张量中，并处理形状转换。

        参数：
            view_tensor (torch.Tensor): 编辑后的视图张量，形状为 [B, C, N, height, width]。
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
        """
        # 将视图张量转换为 [B, N, C, height, width] 以符合 PanoramaTensor 的输入
        view_tensor_reordered = view_tensor.permute(0, 2, 1, 3, 4)
        self.panorama_tensor.set_view_tensor(view_tensor_reordered, fov, theta, phi)

    def set_view_tensor_bilinear(self, view_tensor, fov, theta, phi):
        """
        使用双线性插值将编辑后的视图张量写回等距投影全景张量中，并处理形状转换。

        参数：
            view_tensor (torch.Tensor): 编辑后的视图张量，形状为 [B, C, N, height, width]。
            fov (float): 视野角度（度）。
            theta (float): 水平角度（度，偏航）。
            phi (float): 垂直角度（度，俯仰）。
        """
        # 将视图张量转换为 [B, N, C, height, width] 以符合 PanoramaTensor 的输入
        view_tensor_reordered = view_tensor.permute(0, 2, 1, 3, 4)
        self.panorama_tensor.set_view_tensor_bilinear(view_tensor_reordered, fov, theta, phi)

    def get_equirect_tensor(self):
        """
        获取等距投影全景张量，并将其转换回 [B, C, N, H, W]。

        返回：
            torch.Tensor: 等距投影全景张量，形状为 [B, C, N, H, W]。
        """
        equirect_tensor = self.panorama_tensor.equirect_tensor
        return equirect_tensor.permute(0, 2, 1, 3, 4)

    def set_view_tensor_no_interpolation(self, view_tensor, fov, theta, phi):
        view_tensor_reordered = view_tensor.permute(0, 2, 1, 3, 4)
        self.panorama_tensor.set_view_tensor_no_interpolation(view_tensor_reordered, fov, theta, phi)

