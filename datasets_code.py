from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np

class QuoteCSE_Dataset(Dataset):
    """
    QuoteCSE 모델 학습을 위한 커스텀 데이터셋 클래스
    
    기사의 제목(Title)과 본문 내 여러 문장(Body)을 토큰화하여 저장하고,
    학습 시 배치 단위로 제공하는 역할을 합니다.
    """
    def __init__(self, args, title_texts, body_texts, pos_idx, neg_idx, max_seq):
        """
        데이터셋 초기화 및 전처리 과정을 수행합니다.
        
        Args:
            args: 모델 및 학습 설정을 담은 인자 (tokenizer, max_len 등 포함)
            title_texts (list): 기사 제목 리스트 (Anchor 역할)
            body_texts (list): 기사 본문 내 인용구/문장 리스트들의 리스트 (Positive/Negative Candidates)
            pos_idx (list): 긍정 예시(Positive)의 인덱스 리스트
            neg_idx (list): 부정 예시(Negative)의 인덱스 리스트
            max_seq (int): 하나의 배치 내에서 포함할 최대 문장 개수 (문장 단위 패딩용)
        """
        self.tokenizer = args.tokenizer
        
        # 데이터를 저장할 리스트 초기화
        self.title = []
        self.body = []
        self.pos_idx = []
        self.neg_idx = []
        
        self.max_seq = max_seq      # 문장(Sentence) 개수의 최대 길이 (Zero-padding용)
        self.max_len = args.max_len # 토큰(Token)의 최대 길이 (Truncation용)
        self.body_len = []          # 실제 유효한 문장의 개수를 저장
        
        # 입력 데이터의 개수가 일치하는지 검증
        assert len(title_texts) == len(body_texts) 
        
        print(f"Tokenizing Data... (Total: {len(title_texts)})")
        
        # __init__ 단계에서 미리 토큰화를 수행하여 학습 시 부하를 줄임
        for idx in tqdm(range(len(title_texts))):
            title = title_texts[idx]
            body = body_texts[idx]
            
            # 1. 제목(Title) 토큰화
            # 제목은 단일 문장이므로 일반적인 BERT 입력 형태로 변환
            title_input = self.tokenizer(
                title, 
                padding='max_length', 
                truncation=True,
                max_length=self.max_len, 
                return_tensors='pt'
            )
            
            # 배치 차원(dim=0) 제거 (나중에 DataLoader가 배치 생성 시 다시 묶어줌)
            title_input['input_ids'] = torch.squeeze(title_input['input_ids'])
            title_input['attention_mask'] = torch.squeeze(title_input['attention_mask'])
            
            
            # 2. 본문(Body) 토큰화 및 문장 단위 패딩 처리
            # body는 여러 문장으로 구성된 리스트일 수 있음
            body_input = self.tokenizer(
                body, 
                padding='max_length', 
                truncation=True,
                max_length=self.max_len, 
                return_tensors='pt'
            )
            
            # 실제 문장의 개수를 저장 (나중에 마스킹 등에 활용 가능)
            self.body_len.append(len(body_input['input_ids']))
            
            # [중요] 문장 개수 맞추기 (Zero-Padding for Sentences)
            # 모든 샘플의 문장 개수(max_seq)를 동일하게 맞추기 위해 0으로 채워진 텐서 생성
            # Shape: (최대 문장 개수, 최대 토큰 길이)
            b_input = np.zeros((self.max_seq, self.max_len))
            b_att = np.zeros((self.max_seq, self.max_len))
            b_token = np.zeros((self.max_seq, self.max_len))
            
            # 실제 데이터가 있는 부분만 채워넣음 (나머지는 0으로 유지)
            # 이를 통해 배치 내에서 문장의 개수가 달라도 행렬 연산이 가능해짐
            curr_len = len(body_input['input_ids'])
            b_input[:curr_len] = body_input['input_ids']
            b_att[:curr_len] = body_input['attention_mask']
            b_token[:curr_len] = body_input['token_type_ids']
            
            # Numpy 배열을 PyTorch Tensor로 변환
            b_input = torch.Tensor(b_input)
            b_att = torch.Tensor(b_att)
            b_token = torch.Tensor(b_token)
            
            # 딕셔너리 형태로 저장 및 차원 정리 (squeeze 불필요한 경우 확인 필요하나 여기선 명시적 차원 축소 의도)
            body_input['input_ids'] = torch.squeeze(b_input)
            body_input['attention_mask'] = torch.squeeze(b_att)
            body_input['token_type_ids'] = torch.squeeze(b_token)
            
            # 처리된 데이터를 리스트에 추가
            self.title.append(title_input)
            self.body.append(body_input)
            self.pos_idx.append(pos_idx[idx])
            self.neg_idx.append(neg_idx[idx])
            
    def __len__(self):
        """전체 데이터셋의 길이를 반환"""
        return len(self.title)
    
    def __getitem__(self, idx):
        """
        특정 인덱스(idx)의 데이터를 반환
        
        Returns:
            title_input: 토큰화된 제목 데이터
            body_input: 토큰화 및 패딩된 본문 데이터
            body_len: 실제 본문 문장의 개수
            pos_idx: 긍정 인덱스 (Long Tensor)
            neg_idx: 부정 인덱스 (Long Tensor)
        """
        return (
            self.title[idx], 
            self.body[idx], 
            self.body_len[idx], 
            torch.tensor(self.pos_idx[idx], dtype=torch.long), 
            torch.tensor(self.neg_idx[idx], dtype=torch.long)
        )


def create_data_loader(args, df, shuffle, drop_last):
    """
    DataFrame을 입력받아 PyTorch DataLoader를 생성하는 함수
    
    Args:
        args: 학습 설정 인자
        df (pd.DataFrame): 데이터가 담긴 데이터프레임
        shuffle (bool): 데이터 셔플 여부
        drop_last (bool): 배치를 만들고 남은 데이터를 버릴지 여부
        
    Returns:
        DataLoader: 학습용 데이터 로더
    """
    
    # 데이터셋 인스턴스 생성
    cd = QuoteCSE_Dataset(
        args,
        title_texts=df.title_quote.to_numpy(),       # 제목 컬럼
        body_texts=df.sentence_quotes.to_numpy(),    # 본문(문장 리스트) 컬럼
        pos_idx=df.pos_idx.to_numpy(),               # Positive Index
        neg_idx=df.neg_idx.to_numpy(),               # Negative Index
        # 전체 데이터 중 가장 문장이 많은 경우를 찾아 max_seq로 설정 (동적 패딩)
        max_seq = max(df.sentence_quotes.apply(len).values), 
    )
    
    # DataLoader 반환
    return DataLoader(
        cd,
        batch_size=args.batch_size,
        num_workers=args.num_workers, # 병렬 처리를 위한 워커 수
        shuffle=shuffle,
        drop_last=drop_last
    )


def tuplify_with_device(batch, device):
    """
    DataLoader에서 나온 배치를 GPU(또는 지정된 device)로 이동시키는 헬퍼 함수
    
    Args:
        batch (list/tuple): DataLoader가 반환한 배치의 리스트
        device (torch.device): 데이터를 이동시킬 대상 장치 (cuda or cpu)
        
    Returns:
        tuple: 모델에 입력 가능한 형태의 텐서 튜플
    """
    # batch[0]: Title Input Dictionary
    # batch[1]: Body Input Dictionary
    # batch[2]: (주의) QuoteCSE_Dataset의 __getitem__은 body_len(int)을 반환하므로 
    #           만약 여기서 batch[2]['input_ids']를 호출한다면 
    #           Dataset의 반환값 순서나 구조를 확인해야 합니다.
    #           아래 코드는 3개의 입력 딕셔너리가 있다고 가정한 로직입니다.
    
    return tuple([
        batch[0]['input_ids'].to(device, dtype=torch.long),
        batch[0]['attention_mask'].to(device, dtype=torch.long),
        
        batch[1]['input_ids'].to(device, dtype=torch.long),
        batch[1]['attention_mask'].to(device, dtype=torch.long),
        
        # [Check Point] 아래 batch[2]가 딕셔너리인지 확인 필요 (현재 Dataset은 int 반환)
        batch[2]['input_ids'].to(device, dtype=torch.long), 
        batch[2]['attention_mask'].to(device, dtype=torch.long)
    ])
