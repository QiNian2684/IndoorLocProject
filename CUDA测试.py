"""
CUDA诊断脚本 - 检查CUDA是否可用及潜在问题
"""
import sys
import platform
import subprocess
import os
from datetime import datetime


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_python_info():
    """检查Python环境信息"""
    print_section("Python环境信息")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"处理器: {platform.processor()}")


def check_pytorch():
    """检查PyTorch安装情况"""
    print_section("PyTorch安装状态")

    try:
        import torch
        print(f"✓ PyTorch已安装")
        print(f"  版本: {torch.__version__}")
        print(f"  安装路径: {torch.__file__}")

        # 检查PyTorch编译信息
        print(f"\nPyTorch编译信息:")
        print(f"  CUDA编译版本: {torch.version.cuda if torch.version.cuda else '无(CPU版本)'}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else '不可用'}")

        return True
    except ImportError as e:
        print(f"✗ PyTorch未安装!")
        print(f"  错误: {e}")
        print(f"\n解决方案:")
        print(f"  1. 安装PyTorch:")
        print(f"     pip install torch torchvision torchaudio")
        print(f"  2. 或访问 https://pytorch.org/get-started/locally/ 获取适合您系统的安装命令")
        return False


def check_cuda_availability():
    """检查CUDA可用性"""
    print_section("CUDA可用性检查")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA是否可用: {'✓ 是' if cuda_available else '✗ 否'}")

        if cuda_available:
            print(f"\n✓ CUDA功能正常!")

            # 显示GPU信息
            gpu_count = torch.cuda.device_count()
            print(f"\nGPU数量: {gpu_count}")

            for i in range(gpu_count):
                print(f"\nGPU {i}:")
                print(f"  名称: {torch.cuda.get_device_name(i)}")

                # 显卡属性
                props = torch.cuda.get_device_properties(i)
                print(f"  计算能力: {props.major}.{props.minor}")
                print(f"  总显存: {props.total_memory / 1024 ** 3:.2f} GB")

                # 当前显存使用
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                print(f"  已分配显存: {memory_allocated:.2f} GB")
                print(f"  已预留显存: {memory_reserved:.2f} GB")
                print(f"  可用显存: {(props.total_memory / 1024 ** 3) - memory_reserved:.2f} GB")

            # 测试CUDA运算
            print("\n测试CUDA运算...")
            test_cuda_computation()

        else:
            print("\n✗ CUDA不可用!")
            diagnose_cuda_issues()

        return cuda_available

    except Exception as e:
        print(f"检查CUDA时发生错误: {e}")
        return False


def test_cuda_computation():
    """测试CUDA计算"""
    try:
        import torch
        import time

        # 创建测试张量
        size = 1000
        device = torch.device('cuda')

        # 在GPU上创建张量
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # 预热
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # 测试速度
        start_time = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        # CPU对比测试
        a_cpu = a.cpu()
        b_cpu = b.cpu()

        start_time = time.time()
        for _ in range(100):
            c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time

        print(f"  矩阵乘法测试 ({size}x{size}):")
        print(f"    GPU时间: {gpu_time:.4f} 秒")
        print(f"    CPU时间: {cpu_time:.4f} 秒")
        print(f"    加速比: {cpu_time / gpu_time:.2f}x")
        print(f"\n✓ CUDA计算测试通过!")

    except Exception as e:
        print(f"✗ CUDA计算测试失败: {e}")


def diagnose_cuda_issues():
    """诊断CUDA不可用的原因"""
    print("\n诊断CUDA问题...")

    issues = []
    solutions = []

    # 1. 检查NVIDIA驱动
    print("\n1. 检查NVIDIA驱动...")
    nvidia_driver = check_nvidia_driver()
    if not nvidia_driver:
        issues.append("未检测到NVIDIA驱动")
        solutions.append("安装或更新NVIDIA驱动: https://www.nvidia.com/download/index.aspx")

    # 2. 检查NVIDIA GPU
    print("\n2. 检查NVIDIA GPU...")
    has_gpu = check_nvidia_gpu()
    if not has_gpu:
        issues.append("未检测到NVIDIA GPU")
        solutions.append("确认您的系统有NVIDIA GPU，如果是笔记本，确保未禁用独立显卡")

    # 3. 检查PyTorch版本
    print("\n3. 检查PyTorch CUDA版本...")
    pytorch_cuda = check_pytorch_cuda_version()
    if not pytorch_cuda:
        issues.append("PyTorch是CPU版本，不支持CUDA")
        solutions.append(
            "重新安装CUDA版本的PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # 4. 检查CUDA工具包
    print("\n4. 检查CUDA工具包...")
    cuda_toolkit = check_cuda_toolkit()
    if not cuda_toolkit:
        issues.append("未安装CUDA工具包或版本不匹配")
        solutions.append("安装CUDA工具包: https://developer.nvidia.com/cuda-downloads")

    # 5. 检查环境变量
    print("\n5. 检查环境变量...")
    env_vars = check_environment_variables()
    if not env_vars:
        issues.append("CUDA环境变量未正确设置")
        solutions.append("添加CUDA路径到系统环境变量PATH中")

    # 总结问题
    print_section("诊断结果")

    if issues:
        print("发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\n建议解决方案:")
        for i, solution in enumerate(solutions, 1):
            print(f"  {i}. {solution}")
    else:
        print("未发现明显问题，可能是其他原因导致CUDA不可用")
        print("建议:")
        print("  1. 重启计算机")
        print("  2. 确保NVIDIA GPU未被其他程序占用")
        print("  3. 在设备管理器中检查GPU状态")


def check_nvidia_driver():
    """检查NVIDIA驱动"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # 解析驱动版本
            output = result.stdout
            for line in output.split('\n'):
                if 'Driver Version' in line:
                    print(f"  ✓ NVIDIA驱动已安装")
                    print(f"    {line.strip()}")
                    return True
            return True
        else:
            print(f"  ✗ nvidia-smi命令失败")
            return False
    except FileNotFoundError:
        print(f"  ✗ 未找到nvidia-smi命令")
        return False
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def check_nvidia_gpu():
    """检查是否有NVIDIA GPU"""
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            output = result.stdout.lower()
            if 'nvidia' in output:
                print(f"  ✓ 检测到NVIDIA GPU")
                # 提取GPU名称
                for line in result.stdout.split('\n'):
                    if 'nvidia' in line.lower() and 'name' not in line.lower():
                        print(f"    GPU: {line.strip()}")
                return True
            else:
                print(f"  ✗ 未检测到NVIDIA GPU")
                return False
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")

    # 备用方法
    try:
        import torch
        if hasattr(torch.cuda, 'get_device_name'):
            try:
                name = torch.cuda.get_device_name(0)
                print(f"  ✓ PyTorch检测到GPU: {name}")
                return True
            except:
                pass
    except:
        pass

    return False


def check_pytorch_cuda_version():
    """检查PyTorch是否为CUDA版本"""
    try:
        import torch
        if torch.version.cuda:
            print(f"  ✓ PyTorch是CUDA版本")
            print(f"    CUDA版本: {torch.version.cuda}")
            return True
        else:
            print(f"  ✗ PyTorch是CPU版本")
            return False
    except:
        print(f"  ✗ 无法检查PyTorch版本")
        return False


def check_cuda_toolkit():
    """检查CUDA工具包"""
    try:
        # 尝试运行nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ CUDA工具包已安装")
            # 解析版本
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    print(f"    {line.strip()}")
            return True
        else:
            print(f"  ✗ nvcc命令失败")
            return False
    except FileNotFoundError:
        print(f"  ⚠ 未找到nvcc命令(可能未安装CUDA工具包或未添加到PATH)")
        # 这不一定是问题，因为PyTorch自带CUDA运行时
        return None
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def check_environment_variables():
    """检查环境变量"""
    cuda_paths = []

    # 检查CUDA_PATH
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"  ✓ CUDA_PATH: {cuda_path}")
        cuda_paths.append(cuda_path)
    else:
        print(f"  ⚠ CUDA_PATH未设置")

    # 检查PATH中的CUDA
    path = os.environ.get('PATH', '')
    cuda_in_path = 'cuda' in path.lower()
    if cuda_in_path:
        print(f"  ✓ PATH包含CUDA相关路径")
    else:
        print(f"  ⚠ PATH中未找到CUDA路径")

    return len(cuda_paths) > 0 or cuda_in_path


def generate_report():
    """生成诊断报告"""
    print_section("CUDA诊断报告")
    print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查各项
    check_python_info()

    pytorch_ok = check_pytorch()
    if not pytorch_ok:
        print("\n⚠ PyTorch未安装，无法继续检查CUDA")
        return

    cuda_ok = check_cuda_availability()

    # 最终建议
    print_section("最终建议")

    if cuda_ok:
        print("✓ CUDA工作正常！您可以使用GPU加速。")
        print("\n在代码中使用GPU:")
        print("  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("  model = model.to(device)")
        print("  tensor = tensor.to(device)")
    else:
        print("✗ CUDA当前不可用")
        print("\n快速解决步骤:")
        print("  1. 确认有NVIDIA GPU: 在设备管理器查看")
        print("  2. 更新NVIDIA驱动: https://www.nvidia.com/download/index.aspx")
        print("  3. 重新安装PyTorch CUDA版本:")
        print("     pip uninstall torch torchvision torchaudio")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  4. 重启计算机")
        print("\n如果仍有问题，可以:")
        print("  - 使用CPU版本运行(速度较慢): device = torch.device('cpu')")
        print("  - 使用Google Colab等云服务获取免费GPU")


def main():
    """主函数"""
    print("=" * 60)
    print(" CUDA诊断工具 v1.0")
    print("=" * 60)

    generate_report()

    print("\n" + "=" * 60)
    print(" 诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()