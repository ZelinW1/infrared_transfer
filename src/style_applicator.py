import cv2
import numpy as np
import os
from tqdm import tqdm

def load_maps(fingerprint_dir: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    从指定目录加载相机指纹图（渐晕图和噪声图），格式为 .exr。

    Args:
        fingerprint_dir (str): 存放 .exr 格式特征图的目录。

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: 一个包含 (渐晕图, 噪声图) 的元组。
                                                    如果文件不存在或无法读取，则返回 (None, None)。
    """
    # --- 修改点 1: 文件扩展名从 .tiff 改为 .exr ---
    vignetting_path = os.path.join(fingerprint_dir, 'vignetting_map.exr')
    noise_path = os.path.join(fingerprint_dir, 'noise_map.exr')

    # --- 修改点 2: 更新错误信息 ---
    if not os.path.exists(vignetting_path) or not os.path.exists(noise_path):
        print(f"错误：在 '{fingerprint_dir}' 中找不到所需的 'vignetting_map.exr' 或 'noise_map.exr'。")
        print("请先运行 camera_analyzer.py 来生成这些文件。")
        return None, None

    try:
        # 使用 cv2.IMREAD_UNCHANGED 标志来确保以原始（浮点数）数据类型读取EXR文件
        vignetting_map = cv2.imread(vignetting_path, cv2.IMREAD_UNCHANGED)
        noise_map = cv2.imread(noise_path, cv2.IMREAD_UNCHANGED)
        print("相机指纹图 (EXR) 加载成功。")
        return vignetting_map, noise_map
    except Exception as e:
        print(f"错误：加载EXR指纹图失败: {e}")
        return None, None

def apply_style(
    clean_image: np.ndarray,
    vignetting_map: np.ndarray,
    noise_map: np.ndarray,
    noise_alpha: float = 0.5
) -> np.ndarray:
    """
    将相机指纹（渐晕和噪声）应用到一张干净的图像上。

    Args:
        clean_image (np.ndarray): 输入的干净图像（建议为灰度图）。
        vignetting_map (np.ndarray): 渐晕图（低频）。
        noise_map (np.ndarray): 噪声图（高频）。
        noise_alpha (float): 噪声强度系数，用于控制噪声的明显程度。

    Returns:
        np.ndarray: 添加了相机风格后的图像。
    """
    # 1. 确保输入图像是灰度图
    if len(clean_image.shape) > 2:
        # 如果是彩色图，转换为灰度图
        clean_image_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    else:
        clean_image_gray = clean_image

    # 2. 将特征图的尺寸调整为与干净图像一致
    target_resolution = (clean_image_gray.shape[1], clean_image_gray.shape[0]) # (宽度, 高度)
    
    vignette_resized = cv2.resize(vignetting_map, target_resolution, interpolation=cv2.INTER_CUBIC)
    noise_resized = cv2.resize(noise_map, target_resolution, interpolation=cv2.INTER_CUBIC)

    # 3. 应用渐晕（乘性操作）
    # 首先，对渐晕图进行归一化，使其值分布在 [0, 1] 区间附近，且中心值接近1
    # 这样可以避免图像整体亮度发生剧烈变化
    vignette_normalized = vignette_resized / vignette_resized.max()

    # 将干净图像的数据类型转换为浮点数以进行精确计算
    image_float = clean_image_gray.astype(np.float32)
    
    # 乘法应用渐晕
    image_with_vignette = cv2.multiply(image_float, vignette_normalized)

    # 4. 应用噪声（加性操作）
    # 将带强度系数的噪声图添加到已应用渐晕的图像上
    final_image_float = cv2.add(image_with_vignette, noise_resized * noise_alpha)

    # 5. 后处理：将像素值裁剪到 [0, 255] 范围并转换回 uint8 类型
    # np.clip确保值不会超出范围
    final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

    return final_image

# --- 主程序入口 ---
if __name__ == '__main__':
    print("--- 相机风格应用模块测试 ---")

    # 定义项目的根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 定义路径
    CLEAN_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'data', 'clean_images')
    FINGERPRINT_PATH = os.path.join(PROJECT_ROOT, 'output', 'camera_fingerprint')
    STYLED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'output', 'stylized_images')
    
    # 确保输出目录存在
    os.makedirs(STYLED_OUTPUT_PATH, exist_ok=True)

    # 1. 加载相机指纹
    vignette, noise = load_maps(FINGERPRINT_PATH)

    if vignette is not None and noise is not None:
        # 2. 检查是否有干净的图像需要处理
        try:
            clean_image_files = [
                f for f in os.listdir(CLEAN_IMAGES_PATH)
                if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            ]
        except FileNotFoundError:
            clean_image_files = []

        if not clean_image_files:
            print(f"\n!!! 操作提示 !!!")
            print(f"请将您要添加风格的'干净'图像放入以下目录中:")
            print(f"-> {CLEAN_IMAGES_PATH}")
            os.makedirs(CLEAN_IMAGES_PATH, exist_ok=True)
        else:
            print(f"找到 {len(clean_image_files)} 张干净图像，开始进行风格迁移...")
            # 3. 循环处理每一张干净的图像
            for filename in tqdm(clean_image_files, desc="应用风格中"):
                try:
                    # 读取干净图像
                    clean_img_path = os.path.join(CLEAN_IMAGES_PATH, filename)
                    clean_img = cv2.imread(clean_img_path)

                    if clean_img is None:
                        print(f"\n警告：无法读取图像 {filename}，已跳过。")
                        continue

                    # 应用风格
                    stylized_img = apply_style(
                        clean_image=clean_img,
                        vignetting_map=vignette,
                        noise_map=noise,
                        noise_alpha=0.5 # 这个值可以根据需要调整
                    )

                    # 构建输出文件路径
                    output_filename = f"stylized_{filename}"
                    output_path = os.path.join(STYLED_OUTPUT_PATH, output_filename)
                    
                    # 保存结果
                    cv2.imwrite(output_path, stylized_img)

                except Exception as e:
                    print(f"\n处理文件 {filename} 时发生错误: {e}，已跳过。")
            
            print(f"\n所有图像处理完成！结果已保存至: {STYLED_OUTPUT_PATH}")