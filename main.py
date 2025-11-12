import os
import argparse
import yaml
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from tqdm import tqdm

# 从 src 包中导入我们编写的模块和函数
from src.camera_analyzer import extract_fingerprint, save_maps
from src.style_applicator import load_maps, apply_style

def load_config(config_path='config.yaml') -> dict:
    """加载 YAML 配置文件"""
    print(f"正在从 '{config_path}' 加载配置...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到！请确保该文件存在于项目根目录。")
        exit(1) # 退出程序
    except Exception as e:
        print(f"错误：解析配置文件失败: {e}")
        exit(1)

def run_extraction(config: dict):
    """执行特征提取流程"""
    print("\n--- 步骤 1: 开始提取相机指纹 ---")
    
    # 从配置中获取路径和参数
    paths = config['paths']
    analyzer_params = config['analyzer']
    
    raw_images_dir = paths['raw_images_dir']
    fingerprint_output_dir = os.path.join(paths['output_dir'], paths['fingerprint_subdir'])
    
    # 调用核心函数
    vignette, noise = extract_fingerprint(
        raw_images_dir=raw_images_dir,
        process_resolution=tuple(analyzer_params['process_resolution']),
        blur_kernel_size=tuple(analyzer_params['gaussian_blur_kernel'])
    )
    
    # 保存结果
    if vignette is not None and noise is not None:
        save_maps(vignette, noise, fingerprint_output_dir)
        print("--- 特征提取完成 ---\n")
    else:
        print("--- 特征提取失败 ---\n")
        exit(1) # 提取失败则终止

def run_application(config: dict):
    """执行风格应用流程"""
    print("\n--- 步骤 2: 开始应用相机风格 ---")
    
    # 从配置中获取路径和参数
    paths = config['paths']
    applicator_params = config['applicator']
    
    clean_images_dir = paths['clean_images_dir']
    fingerprint_dir = os.path.join(paths['output_dir'], paths['fingerprint_subdir'])
    stylized_output_dir = os.path.join(paths['output_dir'], paths['stylized_subdir'])
    
    # 确保输出目录存在
    os.makedirs(stylized_output_dir, exist_ok=True)
    
    # 加载指纹
    vignette, noise = load_maps(fingerprint_dir)
    
    if vignette is None or noise is None:
        print("--- 风格应用失败：无法加载指纹图 ---")
        exit(1)
        
    # 获取待处理的图像列表
    try:
        clean_image_files = [f for f in os.listdir(clean_images_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    except FileNotFoundError:
        print(f"错误：干净图像目录 '{clean_images_dir}' 不存在。")
        # 创建目录并提示用户
        os.makedirs(clean_images_dir, exist_ok=True)
        print(f"已创建目录，请将干净图像放入该目录后重试。")
        return

    if not clean_image_files:
        print(f"在 '{clean_images_dir}' 中未找到干净图像。")
        return

    print(f"找到 {len(clean_image_files)} 张干净图像，开始处理...")
    
    # 循环处理
    for filename in tqdm(clean_image_files, desc="应用风格中"):
        try:
            clean_img_path = os.path.join(clean_images_dir, filename)
            clean_img = cv2.imread(clean_img_path)
            
            if clean_img is None:
                continue

            stylized_img = apply_style(
                clean_image=clean_img,
                vignetting_map=vignette,
                noise_map=noise,
                noise_alpha=float(applicator_params['noise_alpha'])
            )
            
            output_path = os.path.join(stylized_output_dir, f"stylized_{filename}")
            cv2.imwrite(output_path, stylized_img)

        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")

    print(f"--- 风格应用完成！结果已保存至: {stylized_output_dir} ---")

def main():
    """主函数：解析参数并执行相应任务"""
    parser = argparse.ArgumentParser(description="红外相机风格迁移工具")
    
    parser.add_argument(
        'action',
        choices=['extract', 'apply', 'all'],
        help="要执行的操作: 'extract' - 仅提取相机指纹; 'apply' - 仅应用风格; 'all' - 执行完整流程。"
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help="指定配置文件的路径 (默认为 'config.yaml')"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 根据用户选择执行不同操作
    if args.action == 'extract':
        run_extraction(config)
    elif args.action == 'apply':
        run_application(config)
    elif args.action == 'all':
        run_extraction(config)
        run_application(config)

if __name__ == '__main__':
    main()