import numpy as np
import torch


class EarlyStopping:
    """早停机制，监控训练过程，避免过拟合"""

    def __init__(self, patience=10, min_delta=0, verbose=True, mode='min', save_path=None):
        """
        初始化早停机制

        参数:
            patience (int): 不改善的轮数，超过此值则停止训练
            min_delta (float): 最小改善阈值，低于此值视为没有改善
            verbose (bool): 是否打印早停信息
            mode (str): 'min'表示监控指标越小越好，'max'表示越大越好
            save_path (str): 保存最佳模型的路径，如果为None则不保存
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, val_loss, model=None):
        """
        每轮训练后调用

        参数:
            val_loss (float): 验证集上的指标
            model: 当前模型，用于保存

        返回:
            bool: 是否应该停止训练
        """
        score = -val_loss if self.mode == 'min' else val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """保存模型检查点"""
        if self.save_path is None or model is None:
            return

        if self.verbose:
            improved = 'decreased' if self.mode == 'min' else 'increased'
            print(f'Validation loss {improved} ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), self.save_path)
        else:
            # 保存非PyTorch模型
            try:
                model.save_model(self.save_path)
            except:
                pass

        self.val_loss_min = val_loss