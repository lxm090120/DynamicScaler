from PIL import Image
import os

def convert_and_compress_image(input_path, output_path, quality=85):
    """
    将 PNG 图片转换为 JPG 格式并压缩，同时保持分辨率不变。

    参数:
    - input_path (str): 输入图片路径。
    - output_path (str): 输出图片路径。
    - quality (int): 图片质量，范围为 1-95，默认值为 85。
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 如果图片是 PNG 格式，转换为 RGB 模式（JPG 不支持透明通道）
            if img.mode in ('RGBA', 'LA'):
                img = img.convert('RGB')
            # 保存为 JPG 格式并调整质量
            img.save(output_path, 'JPEG', quality=quality)
        print(f"图片已成功转换并压缩保存到: {output_path}")
    except Exception as e:
        print(f"处理图片时出错: {e}")

def convert_and_compress_all_images(input_dir, output_dir, quality=80):
    """
    将指定目录中的所有 `.png` 图片文件转换为 `.jpg` 格式并压缩，保存到输出目录。

    参数:
    - input_dir (str): 输入目录路径。
    - output_dir (str): 输出目录路径。
    - quality (int): 图片质量，范围为 1-95，默认值为 80。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 构建完整的输入文件路径
        input_path = os.path.join(input_dir, filename)
        
        # 仅处理 `.png` 文件
        if filename.lower().endswith('.png'):
            # 构建输出文件路径，将扩展名改为 `.jpg`
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            # 转换并压缩图片
            convert_and_compress_image(input_path, output_path, quality=quality)
        else:
            print(f"跳过非 `.png` 文件: {filename}")

# 示例用法
input_dir = "/home/jxliu/test/FIFO-Diffusion_public/GIF"  # 输入目录路径
output_dir = "/home/jxliu/test/FIFO-Diffusion_public/New_GIF"  # 输出目录路径
convert_and_compress_all_images(input_dir, output_dir, quality=80)  # 压缩质量设置为 33