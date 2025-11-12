import cv2
import numpy as np
import os
from tqdm import tqdm

# 定义支持的图像文件扩展名，避免读取无关文件
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

def extract_fingerprint(
    raw_images_dir: str,
    process_resolution: tuple = (512, 512),
    blur_kernel_size: tuple = (99, 99)
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    分析一个目录中的所有图像，以提取相机的“指纹”（渐晕图和噪声图）。

    核心流程:
    1. 读取目录下所有图像。
    2. 将每张图像统一尺寸、转为灰度并累加。
    3. 计算所有图像的像素平均值，得到“平均图像”。
    4. 对平均图像进行高斯模糊得到低频的“渐晕图”。
    5. 从平均图像中减去渐晕图，得到高频的“噪声图”。

    Args:
        raw_images_dir (str): 存放原始红外图像的目录路径。
        process_resolution (tuple): 用于分析的图像处理分辨率 (宽度, 高度)。
        blur_kernel_size (tuple): 用于分离高低频的高斯模糊核大小 (必须是奇数)。

    Returns:
        tuple[np.ndarray, np.ndarray]: 一个包含 (渐晕图, 噪声图) 的元组。
                                       如果找不到或无法处理任何图像，则返回 (None, None)。
    """
    # 1. 获取所有支持的图像文件的路径
    try:
        image_paths = [
            os.path.join(raw_images_dir, f)
            for f in os.listdir(raw_images_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
    except FileNotFoundError:
        print(f"错误：目录不存在 -> '{raw_images_dir}'")
        return None, None

    if not image_paths:
        print(f"错误：在目录 '{raw_images_dir}' 中没有找到任何支持的图像文件。")
        return None, None

    print(f"找到 {len(image_paths)} 张图像，开始处理...")

    # 2. 初始化一个累加器，用于存放所有图像的总和
    # 使用 float64 类型以保证足够的精度，防止像素值在累加过程中溢出
    accumulator = np.zeros((process_resolution[1], process_resolution[0]), dtype=np.float64)
    processed_image_count = 0

    # 3. 遍历、读取、处理并累加所有图像
    for path in tqdm(image_paths, desc="分析图像中"):
        try:
            # 直接以灰度模式读取图像，提高效率
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"\n警告：无法读取图像 {os.path.basename(path)}，已跳过。")
                continue

            # 将图像尺寸统一到指定的处理分辨率
            img_resized = cv2.resize(img, process_resolution, interpolation=cv2.INTER_AREA)

            # 将调整尺寸后的图像（类型为 uint8）转换为 float64 并加到累加器上
            accumulator += img_resized.astype(np.float64)
            processed_image_count += 1

        except Exception as e:
            print(f"\n警告：处理文件 {os.path.basename(path)} 时发生错误: {e}，已跳过。")
            continue

    if processed_image_count == 0:
        print("错误：没有任何图像被成功处理，无法提取特征。")
        return None, None

    # 4. 计算平均图像
    print(f"\n处理完成 {processed_image_count}/{len(image_paths)} 张图像。正在计算平均图像...")
    average_image = accumulator / processed_image_count

    # 5. 分离低频和高频成分
    print("正在分离低频（渐晕）和高频（噪声）成分...")
    # 低频成分（渐晕图）通过对平均图像进行一次强力的高斯模糊获得
    vignetting_map = cv2.GaussianBlur(average_image, blur_kernel_size, 0)

    # 高频成分（噪声图）是平均图像与低频成分的差
    noise_map = average_image - vignetting_map

    print("相机指纹提取成功！")
    return vignetting_map, noise_map


def save_maps(
    vignetting_map: np.ndarray,
    noise_map: np.ndarray,
    output_dir: str
):
    """
    将提取出的特征图保存为高精度的 32位 EXR 文件。

    Args:
        vignetting_map (np.ndarray): 渐晕图。
        noise_map (np.ndarray): 噪声图。
        output_dir (str): 保存特征图的目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    vignetting_path = os.path.join(output_dir, 'vignetting_map.exr')
    noise_path = os.path.join(output_dir, 'noise_map.exr')
    vignetting_normalized_path = os.path.join(output_dir, 'vignetting_map_normalized.exr')

    # 归一化渐晕图并保存
    normalized_vignette = vignetting_map / vignetting_map.max()

    try:
        # 将NumPy数组的数据类型转换为 32位浮点数 (float32)
        cv2.imwrite(vignetting_path, vignetting_map.astype(np.float32))
        cv2.imwrite(noise_path, noise_map.astype(np.float32))
        cv2.imwrite(vignetting_normalized_path, normalized_vignette.astype(np.float32))

        # --- 修改点 2: 更新打印信息 ---
        print(f"渐晕图已保存至 (EXR格式): {vignetting_path}")
        print(f"噪声图已保存至 (EXR格式): {noise_path}")
        print(f"归一化渐晕图已保存至 (EXR格式): {vignetting_normalized_path}")

    except Exception as e:
        print(f"错误：保存EXR文件失败。请确保您的OpenCV版本支持EXR。错误信息: {e}")


# --- 主程序入口 ---
# 只有当这个脚本被直接执行时，下面的代码块才会运行
# 如果它被其他脚本作为模块导入，则不会运行
if __name__ == '__main__':
    print("--- 相机特征分析模块测试 ---")

    # 定义项目的根目录（假设此脚本在 'src' 文件夹下）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 根据我们之前设计的项目结构，定义输入和输出路径
    RAW_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw_images')
    OUTPUT_FINGERPRINT_PATH = os.path.join(PROJECT_ROOT, 'output', 'camera_fingerprint')

    # 检查原始图像目录是否存在且不为空
    if not os.path.isdir(RAW_IMAGES_PATH) or not os.listdir(RAW_IMAGES_PATH):
        print(f"\n!!! 操作提示 !!!")
        print(f"请先将您要分析的红外图像放入以下目录中:")
        print(f"-> {RAW_IMAGES_PATH}")
        # 如果目录不存在，则创建它
        os.makedirs(RAW_IMAGES_PATH, exist_ok=True)
    else:
        # --- 核心执行流程 ---
        # 1. 提取相机指纹
        vignette, noise = extract_fingerprint(
            raw_images_dir=RAW_IMAGES_PATH,
            process_resolution=(512, 512), # 可根据图像平均尺寸调整
            blur_kernel_size=(99, 99)      # 必须是大的奇数
        )

        # 2. 如果提取成功，则保存结果
        if vignette is not None and noise is not None:
            save_maps(vignette, noise, OUTPUT_FINGERPRINT_PATH)