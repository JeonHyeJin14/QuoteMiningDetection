import random
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """평균값 계산 유틸"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_backbone_and_tokenizer(args):
    """모델 타입에 따른 백본 로드"""
    # Hugging Face 모델 로드 (roberta-base 등)
    if args.backbone == "hf":
        if args.hf_model_name is None:
            raise ValueError("--hf_model_name required for HF backbone")
        
        print(f"[INFO] Using HF backbone: {args.hf_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        backbone = AutoModel.from_pretrained(args.hf_model_name)
        return backbone, tokenizer

    # KoBERT (혹시 나중에 필요할까봐 남겨둠, 기본값은 아님)
    elif args.backbone == "kobert":
        try:
            from kobert_transformers import get_kobert_model, get_tokenizer
            print("[INFO] Using KoBERT backbone")
            backbone = get_kobert_model()
            tokenizer = get_tokenizer()
            return backbone, tokenizer
        except ImportError:
             raise ImportError("KoBERT library not found.")
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
