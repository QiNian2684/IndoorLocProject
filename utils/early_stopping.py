import numpy as np
import torch


class EarlyStopping:
    """��ͣ���ƣ����ѵ�����̣���������"""

    def __init__(self, patience=10, min_delta=0, verbose=True, mode='min', save_path=None):
        """
        ��ʼ����ͣ����

        ����:
            patience (int): �����Ƶ�������������ֵ��ֹͣѵ��
            min_delta (float): ��С������ֵ�����ڴ�ֵ��Ϊû�и���
            verbose (bool): �Ƿ��ӡ��ͣ��Ϣ
            mode (str): 'min'��ʾ���ָ��ԽСԽ�ã�'max'��ʾԽ��Խ��
            save_path (str): �������ģ�͵�·�������ΪNone�򲻱���
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
        ÿ��ѵ�������

        ����:
            val_loss (float): ��֤���ϵ�ָ��
            model: ��ǰģ�ͣ����ڱ���

        ����:
            bool: �Ƿ�Ӧ��ֹͣѵ��
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
        """����ģ�ͼ���"""
        if self.save_path is None or model is None:
            return

        if self.verbose:
            improved = 'decreased' if self.mode == 'min' else 'increased'
            print(f'Validation loss {improved} ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), self.save_path)
        else:
            # �����PyTorchģ��
            try:
                model.save_model(self.save_path)
            except:
                pass

        self.val_loss_min = val_loss