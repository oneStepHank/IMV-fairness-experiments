import torch
import torch.nn as nn

class MedicalFTTransformer(nn.Module):
    def __init__(self, num_numerical=65, cat_categories=[6, 6], embed_dim=32):
        super().__init__()
        
        # 1. 수치형 피처 토크나이저: 65개 피처 각각을 embed_dim 차원으로 투영
        self.num_tokenizers = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_numerical)
        ])
        
        # 2. 범주형 피처 토크나이저: ETHNICITY, INSURANCE 임베딩
        self.cat_tokenizers = nn.ModuleList([
            nn.Embedding(cat_count, embed_dim) for cat_count in cat_categories
        ])
        
        # 3. CLS 토큰 (전체 정보를 요약하여 예측에 사용될 특수 토큰)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 5. Prediction Head (MLP)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # 회귀라면 1, 이진 분류라면 1(sigmoid) 혹은 2
        )

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        
        # 수치형 임베딩: [batch, 65, embed_dim]
        num_tokens = torch.stack([
            emb(x_num[:, i].unsqueeze(1)) for i, emb in enumerate(self.num_tokenizers)
        ], dim=1)
        
        # 범주형 임베딩: [batch, 2, embed_dim]
        cat_tokens = torch.stack([
            emb(x_cat[:, i]) for i, emb in enumerate(self.cat_tokenizers)
        ], dim=1)
        
        # 모든 토큰 결합 + CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, num_tokens, cat_tokens], dim=1) # [batch, 68, embed_dim]
        
        # Transformer 통과
        x = self.transformer(x)
        
        # CLS 토큰의 결과값만 추출하여 예측
        return self.head(x[:, 0, :])