import torch.nn as nn

class FramingClassifier(nn.Module):
    """
    RoBERTa-base Backbone + Classification Head 구조의 모델
    """
    def __init__(self, backbone, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.backbone = backbone
        
        # Backbone의 Hidden Size 자동 감지 (roberta-base는 768)
        if hasattr(backbone.config, 'hidden_size'):
            self.hidden_size = backbone.config.hidden_size
        else:
            self.hidden_size = 768 
            
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Backbone Forward
        # token_type_ids 지원 여부 체크 후 전달
        if token_type_ids is not None and "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooling Strategy
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_token = outputs.pooler_output
        else:
            cls_token = outputs.last_hidden_state[:, 0, :]
            
        x = self.dropout(cls_token)
        logits = self.classifier(x)
        
        return logits
