import numpy as np
import cv2

def prepare_image(
    img0: np.ndarray, 
    target_size: tuple = (960, 544), 
    keep_aspect: bool = True
) -> np.ndarray:
    """
    预处理图像，缩放到目标尺寸，可选择保持宽高比
    
    Args:
        img0: 输入的BGR图像 (H,W,C)
        target_size: 目标尺寸 (width, height)
        keep_aspect: 是否保持原始宽高比
        
    Returns:
        处理后的BGR图像 (target_height, target_width, C)
    """
    # 参数检查
    if not isinstance(img0, np.ndarray) or len(img0.shape) != 3:
        raise ValueError("输入必须是3通道的BGR图像")
        
    if img0.shape[2] != 3:
        raise ValueError("输入图像必须有3个通道(BGR)")
        
    target_width, target_height = target_size
    
    # 不保持宽高比的情况下直接缩放
    if not keep_aspect:
        return cv2.resize(img0, (target_width, target_height))
    
    # 保持宽高比的缩放
    h, w = img0.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized_img = cv2.resize(img0, (new_w, new_h))

    ## TODO: 使用 cv2.copyMakeBorder 效率更高
    
    # 创建目标图像并用黑色填充
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    ## 复制到右下角
    x_offset = 0
    y_offset = 0

    # 将缩放后的图像放入目标图像右下角
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return result