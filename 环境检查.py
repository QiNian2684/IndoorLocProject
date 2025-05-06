import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    # 测试GPU是否工作正常
    x = torch.rand(5, 3).cuda()
    print(f"测试张量在GPU上: {x.device}")