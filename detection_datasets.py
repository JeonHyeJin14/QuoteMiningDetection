from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch
import numpy as np
from util import most_sim

class Contextomized_Detection_Dataset(Dataset):
    """
    탐지 모델(Detection Model)을 위한 데이터셋 클래스
    
    기사 제목(Title)과 본문(Body)을 입력받아 BERT류 모델에 넣을 수 있도록 
    토큰화(Tokenization) 및 패딩(Padding) 처리를 수행합니다.
    """
    def __init__(self, args, title_texts, body_texts, label, max_seq=85):
        """
        Args:
            args: 모델 설정값 (tokenizer, max_len 등)
            title_texts: 기사 제목 리스트
            body_texts: 기사 본문 리스트 (문장 단위로 분리된 리스트의 리스트 형태 권장)
            label: 정답 레이블 (0 or 1)
            max_seq: 본문 내 최대 문장 개수 (문장 단위 패딩을 위해 사용)
        """
        self.tokenizer = args.tokenizer
        self.title = []
        self.body = []
        self.label = []
        
        self.max_seq = max_seq      # 문장 개수(Sentence Count)의 최대 한계
        self.max_len = args.max_len # 토큰 길이(Token Length)의 최대 한계
        self.body_len = []          # 각 샘플의 실제 유효 문장 개수를 저장
        
        # 데이터 개수 무결성 확인
        assert len(title_texts) == len(body_texts) 
        
        print("Tokenizing data... (Preprocessing)")
        for idx in tqdm(range(len(title_texts))):
            title = title_texts[idx]
            body = body_texts[idx]
            
            # -------------------------------------------------------
            # 1. Title Tokenization (제목 전처리)
            # -------------------------------------------------------
            title_input = self.tokenizer(
                title, 
                padding='max_length', 
                truncation=True,
                max_length=self.max_len, 
                return_tensors='pt'
            )
            # 배치 차원 제거 (DataLoader에서 다시 묶으므로 1차원으로 유지)
            title_input['input_ids'] = torch.squeeze(title_input['input_ids'])
            title_input['attention_mask'] = torch.squeeze(title_input['attention_mask'])
            
            # 모델 호환성: BERT는 token_type_ids가 필요하고, RoBERTa는 필요 없음
            if 'token_type_ids' in title_input:
                 title_input['token_type_ids'] = torch.squeeze(title_input['token_type_ids'])
            
            
            # -------------------------------------------------------
            # 2. Body Tokenization (본문 전처리)
            # -------------------------------------------------------
            # 본문은 여러 문장으로 구성되어 있으므로, 각각 토큰화 후 하나의 텐서로 병합 준비
            body_input = self.tokenizer(
                body, 
                padding='max_length', 
                truncation=True,
                max_length=self.max_len, 
                return_tensors='pt'
            )
            
            # 실제 유효한 문장의 개수 저장 (나중에 패딩 제거 시 사용)
            self.body_len.append(len(body_input['input_ids']))
            
            # [Zero-Padding Matrix 초기화]
            # 배치 처리를 위해 모든 샘플을 (max_seq, max_len) 크기로 고정해야 함
            b_input = np.zeros((self.max_seq, self.max_len))
            b_att = np.zeros((self.max_seq, self.max_len))
            b_token = np.zeros((self.max_seq, self.max_len))
            
            # 실제 데이터가 존재하는 부분만 덮어쓰기 (나머지는 0으로 남음)
            curr_len = len(body_input['input_ids'])
            b_input[:curr_len] = body_input['input_ids']
            b_att[:curr_len] = body_input['attention_mask']
            
            # token_type_ids가 존재하는 경우에만 처리
            if 'token_type_ids' in body_input:
                b_token[:curr_len] = body_input['token_type_ids']
            
            # Numpy -> Tensor 변환
            b_input = torch.Tensor(b_input)
            b_att = torch.Tensor(b_att)
            b_token = torch.Tensor(b_token) 
            
            # 딕셔너리에 저장
            body_input['input_ids'] = b_input
            body_input['attention_mask'] = b_att
            body_input['token_type_ids'] = b_token
            
            self.title.append(title_input)
            self.body.append(body_input)
            self.label.append(label[idx])
            
    def __len__(self):
        return len(self.title)
    
    def __getitem__(self, idx):
        """
        DataLoader가 호출할 때 하나의 샘플을 반환
        return: (제목, 본문, 본문_유효_길이, 정답_레이블)
        """
        return (
            self.title[idx], 
            self.body[idx], 
            self.body_len[idx], 
            torch.tensor(self.label[idx], dtype=torch.long)
        )


# ============================================================
# DataLoader 생성 함수
# ============================================================

def create_data_loader(args, df, shuffle, drop_last):
    """
    Pandas DataFrame을 받아 학습 가능한 DataLoader로 변환하는 함수
    """
    # DataFrame에서 필요한 텍스트와 레이블 추출
    # (주의: article_text를 title과 body 양쪽에 사용 중. 의도된 것인지 확인 필요)
    title_texts = df.article_text.to_numpy()    
    body_texts = df.article_text.to_numpy()     
    labels = df.label.to_numpy()
    
    # max_seq 설정 (데이터셋의 가장 긴 문단 길이에 맞추거나 고정값 사용)
    max_seq = 85 
    
    # Custom Dataset 인스턴스 생성
    cd = Contextomized_Detection_Dataset(
        args, 
        title_texts=title_texts,
        body_texts=body_texts,
        label=labels,
        max_seq=max_seq,
    )

    # PyTorch DataLoader 생성
    loader = DataLoader(
        cd,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
    )
    return loader


def make_tensorloader(args, encoder, data_loader, train=False):
    """
    [핵심 기능: Feature Caching]
    기존 DataLoader(Raw Text)를 돌며 Encoder(BERT 등)를 통과시켜,
    '텍스트'가 아닌 '임베딩 벡터'와 '유사도 특징'으로 변환된 새로운 Loader를 생성합니다.
    
    -> 이렇게 하면 분류기(Classifier) 학습 시 매번 무거운 BERT를 돌리지 않아 속도가 빨라집니다.
    
    Args:
        encoder: 학습된 언어 모델 (e.g., BERT/RoBERTa)
        data_loader: 텍스트가 담긴 원본 DataLoader
    """
    output = []
    labels = []
    
    # 인코더는 평가 모드로 고정 (Gradient 계산 X)
    encoder.eval()
    
    with torch.no_grad():
        for title, body, body_len, label in tqdm(data_loader, desc="Encoding & Feature Extraction"):
            
            # 1. Title Data GPU 이동
            title_id = title['input_ids'].to(args.device).long()
            title_at = title['attention_mask'].to(args.device).long()
            
            # 2. Body Data GPU 이동 및 유효 구간 슬라이싱
            # (Zero-Padding된 부분을 제외하고 실제 문장만 인코더에 넣기 위함)
            b_ids = []
            b_atts = []
            
            # 배치 내의 각 샘플(b)에 대해 반복
            for b in range(len(body_len)):
                i = body_len[b] # 해당 샘플의 실제 문장 개수
                
                # 유효한 문장(0~i)까지만 잘라서 가져옴 (메모리 절약 및 연산 효율화)
                b_id = body['input_ids'][b][:i].to(args.device).long()
                b_at = body['attention_mask'][b][:i].to(args.device).long()
                
                b_ids.append(b_id)
                b_atts.append(b_at)
            
            # 효율적인 연산을 위해 배치 내 모든 문장을 하나의 긴 텐서로 병합
            body_ids = torch.cat(b_ids, dim=0)
            body_atts = torch.cat(b_atts, dim=0)

            # 3. Encoder Forward (임베딩 추출)
            # Title과 Body 문장들을 한 번에 모델에 태움
            outs = encoder(
                input_ids = torch.cat([title_id, body_ids]), 
                attention_mask = torch.cat([title_at, body_atts]),
            )

            # 4. Similarity Calculation (유사도 계산)
            # util.most_sim 함수가 Title과 각 Body 문장 간의 유사도를 계산하여
            # 가장 관련성 높은 문장의 특징(s1, s2)을 반환한다고 가정
            s1, s2 = most_sim(outs, args.batch_size, body_len)

            # 5. Feature Engineering (NLI 스타일 특징 결합)
            # (u, v) -> (u, v, |u-v|, u*v) 형태로 결합하여 분류 성능 향상
            s = torch.cat([s1, s2, abs(s1-s2), s1*s2], dim=1)
            
            output.append(s)
            labels.append(label)
            
        # 리스트에 모인 텐서들을 하나의 큰 텐서로 통합
        output = torch.cat(output, dim=0).contiguous().squeeze()
        labels = torch.cat(labels)

    # 6. 최종 TensorDataset 생성
    # 이제 텍스트가 아닌 '특징 벡터(Output)'와 '정답(Label)'만 존재
    linear_ds = TensorDataset(output, labels)
    linear_loader = DataLoader(linear_ds, batch_size=args.batch_size, shuffle=train, drop_last=True)
    
    return linear_loader
