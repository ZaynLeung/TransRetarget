import torch
import torch.nn as nn
from model.model_poseformer import PoseTransformer
from config.variables_define import *
def check_model_parameters(model):
    """
    检查模型参数是否有异常
    """
    print("\n=== 模型参数检查 ===")
    
    total_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0
    large_params = 0
    small_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 检查参数的统计信息
        param_mean = param.data.mean().item()
        param_std = param.data.std().item()
        param_min = param.data.min().item()
        param_max = param.data.max().item()
        
        # 检查异常值
        nan_count = torch.isnan(param.data).sum().item()
        inf_count = torch.isinf(param.data).sum().item()
        zero_count = (param.data == 0).sum().item()
        large_count = (torch.abs(param.data) > 100).sum().item()
        small_count = (torch.abs(param.data) < 1e-8).sum().item()
        
        nan_params += nan_count
        inf_params += inf_count
        zero_params += zero_count
        large_params += large_count
        small_params += small_count
        
        print(f"Layer: {name}")
        print(f"  Shape: {list(param.shape)}, Params: {param.numel()}")
        print(f"  Mean: {param_mean:.6f}, Std: {param_std:.6f}")
        print(f"  Min: {param_min:.6f}, Max: {param_max:.6f}")
        print(f"  NaN: {nan_count}, Inf: {inf_count}, Zero: {zero_count}")
        print(f"  Large(>100): {large_count}, Small(<1e-8): {small_count}")
        print()
    
    print(f"=== 总体参数统计 ===")
    print(f"总参数数量: {total_params}")
    print(f"零参数数量: {zero_params} ({zero_params/total_params*100:.2f}%)")
    print(f"NaN参数数量: {nan_params} ({nan_params/total_params*100:.2f}%)")
    print(f"Infinity参数数量: {inf_params} ({inf_params/total_params*100:.2f}%)")
    print(f"大参数数量(>100): {large_params} ({large_params/total_params*100:.2f}%)")
    print(f"小参数数量(<1e-8): {small_params} ({small_params/total_params*100:.2f}%)")
    
    # 检查是否存在异常
    has_issues = False
    if nan_params > 0:
        print("⚠️  发现 NaN 参数!")
        has_issues = True
    if inf_params > 0:
        print("⚠️  发现 Infinity 参数!")
        has_issues = True
    if large_params / total_params > 0.01:  # 超过1%
        print("⚠️  发现过多的大参数值!")
        has_issues = True
    if zero_params / total_params > 0.5:  # 超过50%
        print("⚠️  发现过多的零参数值!")
        has_issues = True
    
    if not has_issues:
        print("✅ 模型参数检查通过，未发现明显异常")
    
    return not has_issues

def check_model_loading(model_path=None):
    """
    检查加载的模型参数
    """
    if model_path is None:
        model_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\测试用\\model_final.pth"
    
    # 定义模型参数
    
    try:
        # 创建模型实例
        model = PoseTransformer(
            num_frame=receptive_field, 
            in_num_joints=num_joints, 
            in_chans=3, 
            out_num_joint=out_num_joint, 
            out_chans=1,
            embed_dim_ratio=embed_dim_ratio, 
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            spatial_mlp_ratio=spatial_mlp_ratio,
            temporal_mlp_ratio=temporal_mlp_ratio, 
            num_heads=num_heads,
            qkv_bias=True, 
            qk_scale=None,
            drop_path_rate=drop_path_rate,
            angle_limit_rad=angle_limit_rob
        )

        # 如果有可用的GPU，将模型移到GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 如果使用了DataParallel，在加载权重之前需要先转换回来
        if isinstance(model, nn.DataParallel):
            model = model.module

        # 加载模型权重
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 如果检查点是完整字典（包含epoch, model_pos等信息）
        if isinstance(checkpoint, dict) and 'model_pos' in checkpoint:
            model.load_state_dict(checkpoint['model_pos'], strict=False)
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # 直接是state_dict的情况
            model.load_state_dict(checkpoint, strict=False)
        
        print("Model loaded successfully!")
        print(f"Model is on device: {next(model.parameters()).device}")
        
        # 检查模型参数
        params_ok = check_model_parameters(model)
        
        return model, params_ok
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None, False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

if __name__ == "__main__":
    model, params_ok = check_model_loading(model_path='D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\测试用\\model_final.pth')
    if model is not None:
        print(f"Model parameter check completed. Parameters OK: {params_ok}")
    else:
        print("Model parameter check failed.")