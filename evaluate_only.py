import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 설정 (사용자가 실행한 명령어 기준)
# ==========================================
# ★ 여기를 방금 실행한 명령어에 맞췄습니다.
DATA_PATH = "/content/merge_data_labeled.csv"
MODEL_PATH = "/content/QuoteMiningDetection/model/framing_classifier/classifier_best.bin"

# ==========================================
# 2. Dataset & Model Class (train_classifier.py와 동일)
# ==========================================
class FramingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        # distorted 컬럼이 있으면 pair로 처리
        self.use_pair = "distorted" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article = str(row["article_text"])
        label = int(row["label"])
        
        if self.use_pair:
            original = str(row["distorted"])
            encoding = self.tokenizer(
                original, 
                article, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_len, 
                return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                article, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_len, 
                return_tensors="pt"
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return item

class FramingClassifier(nn.Module):
    def __init__(self, backbone_name="roberta-base", num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        # train_classifier.py 로직에 맞춤
        self.hidden_size = 768 
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output이 있으면 쓰고 없으면 cls token 사용
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            x = outputs.pooler_output
        else:
            x = outputs.last_hidden_state[:, 0, :]
            
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# ==========================================
# 3. 평가 실행
# ==========================================
def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loading Model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"!!! 오류: 모델 파일이 {MODEL_PATH}에 없습니다.")
        return

    # --- 데이터 로드 ---
    print("[INFO] Loading Data & Tokenizer...")
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_pickle(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Test Set 분리 (학습 때 사용한 split_seed=0 기준일 가능성 높음, 기본값 0으로 설정)
    # 만약 학습때 --split_seed를 따로 줬다면 그 값으로 바꿔야 합니다. (기본값 0 가정)
    _, df_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df['label'])
    
    test_dataset = FramingDataset(df_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # --- 모델 로드 ---
    # 초기화 (껍데기)
    model = FramingClassifier(backbone_name="roberta-base")
    
    # 저장된 가중치 로드
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    
    # --- 예측 수행 ---
    print(f"[INFO] Running Inference on {len(df_test)} samples...")
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs[:, 1].cpu().tolist()) # Class 1 확률
            
    # --- 결과 계산 ---
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
        
    # --- 결과 출력 ---
    print("\n" + "="*40)
    print("===== Final Evaluation Result =====")
    print("="*40)
    print(f"accuracy        : {acc:.4f}")
    print(f"f1_macro        : {f1_macro:.4f}")
    print(f"auc_score       : {auc:.4f}")
    
    print("\n----- Per-class metrics -----")
    label_ids = [0, 1]
    
    prec_per_class = precision_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    
    for i in label_ids:
        print(f"[{i}] Class {i}")
        print(f"  precision: {prec_per_class[i]:.4f}")
        print(f"  recall   : {rec_per_class[i]:.4f}")
        print(f"  f1       : {f1_per_class[i]:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()