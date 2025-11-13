import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, backbone, hidden_size=100):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size

        backbone_dim = backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        z = self.projection(pooled)
        return F.normalize(z, dim=-1)
    
    
    
class Detection_Model(nn.Module):
    def __init__(self, num_cls, args):
        super(Detection_Model, self).__init__()
        self.dim = args.classifier_input_size*4
        self.hidden = args.classifier_hidden_size
        self.out = nn.Sequential(nn.Linear(self.dim, self.hidden),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, num_cls, bias=True))

    def forward(self, embedding):
        return self.out(embedding)
