import random
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def set_seed(seed):
    """
    실험의 재현성(Reproducibility)을 보장하기 위해 모든 난수 생성기의 시드(Seed)를 고정합니다.
    
    Args:
        seed (int): 고정할 시드 값 (예: 42)
    
    Why:
        - 딥러닝 학습은 초기화, 데이터 셔플링, Dropout 등에서 무작위성을 가집니다.
        - 시드를 고정하지 않으면 같은 코드를 돌려도 매번 결과가 달라져 성능 개선 여부를 판단하기 어렵습니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Multi-GPU 환경까지 고려
        
    # 추가적인 결정론적(Deterministic) 옵션 (속도는 느려질 수 있으나 재현성은 완벽해짐)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random Seed Set to: {seed}")
    
class AverageMeter(object):
    """
    학습 중 발생하는 수치(Loss, Accuracy 등)의 평균을 실시간으로 계산하고 저장하는 유틸리티 클래스입니다.
    
    사용 예:
        배치마다 변하는 Loss 값을 누적하여, 1 Epoch가 끝났을 때의 정확한 평균 Loss를 구할 때 사용합니다.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0   # 현재 스텝의 값
        self.avg = 0   # 누적 평균 값
        self.sum = 0   # 값의 총합
        self.count = 0 # 업데이트 횟수(샘플 수)
        
    def update(self, val, n=1):
        """
        새로운 값(val)을 받아 평균을 갱신합니다.
        Args:
            val: 측정된 값 (예: batch loss)
            n: 해당 값에 기여한 샘플 수 (예: batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_backbone_and_tokenizer(args):
    """
    인자(args)에 따라 적절한 Pre-trained 모델(Backbone)과 Tokenizer를 로드합니다.
    
    특징:
        - 'hf' 모드: HuggingFace Hub에 있는 모든 모델(BERT, RoBERTa, DeBERTa 등)을 지원합니다.
        - 'kobert' 모드: SKT KoBERT와 같은 특수 라이브러리 의존 모델을 위한 확장성을 남겨두었습니다.
    """
    # ---------------------------------------------------------
    # Case 1: Hugging Face Transformers (Standard)
    # ---------------------------------------------------------
    if args.backbone == "hf":
        if args.hf_model_name is None:
            raise ValueError("--hf_model_name required for HF backbone")
        
        print(f"[INFO] Using HF backbone: {args.hf_model_name}")
        
        # AutoTokenizer/AutoModel을 사용하여 모델명만 바꾸면 자동으로 구조가 변경되도록 설계
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        backbone = AutoModel.from_pretrained(args.hf_model_name)
        return backbone, tokenizer

    # ---------------------------------------------------------
    # Case 2: KoBERT (Legacy Support)
    # ---------------------------------------------------------
    elif args.backbone == "kobert":
        try:
            from kobert_transformers import get_kobert_model, get_tokenizer
            print("[INFO] Using KoBERT backbone")
            backbone = get_kobert_model()
            tokenizer = get_tokenizer()
            return backbone, tokenizer
        except ImportError:
             # 라이브러리가 설치되지 않은 환경을 대비한 에러 메시지
             raise ImportError("KoBERT 라이브러리가 설치되지 않았습니다. (pip install kobert-transformers)")
    else:
        raise ValueError(f"지원하지 않는 Backbone 타입입니다: {args.backbone}")
