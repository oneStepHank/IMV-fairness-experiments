import torch
import torch.nn as nn
import rtdl
import pandas as pd
import numpy as np
from loss import FocalLoss
from evalutate import evaluate_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


# helper function
def extract_values(df, num_cols, cat_cols, target_col):
    X_num = df[num_cols].values
    X_cat = df[cat_cols].values
    y = df[target_col].values
    return X_num, X_cat, y

# (3) 텐서 변환 (타입 중요!)
def to_tensor(X_n, X_c, y):
    return (
        torch.tensor(X_n, dtype=torch.float32),
        torch.tensor(X_c.astype(int), dtype=torch.long), # 반드시 long
        torch.tensor(y, dtype=torch.long)
    )
    
# ==========================================
# 1. 데이터 로드 (이미 나누어진 파일 로드)
# ==========================================
DATA_PATH = "/home/kogmaw12/AIEC/_HG/IVM-fairness-transformer/data/final_split"
train_df = pd.read_csv(DATA_PATH+"/train.csv")
test_df = pd.read_csv(DATA_PATH+"/test.csv")
val_df = pd.read_csv(DATA_PATH+"/val.csv")

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

target_col = 'hospital_mortality'
cat_cols = ['ETHNICITY']
unused_cols = ['SUBJECT_ID', 'INSURANCE']

num_cols = [c for c in train_df.columns if c not in cat_cols + [target_col] + unused_cols]

# Raw 데이터 추출
X_num_train, X_cat_train, y_train = extract_values(train_df, num_cols, cat_cols, target_col)
X_num_val,   X_cat_val,   y_val   = extract_values(val_df, num_cols, cat_cols, target_col)
X_num_test,  X_cat_test,  y_test  = extract_values(test_df, num_cols, cat_cols, target_col)

# ==========================================
# 3. 전처리 (Fit on Train, Transform on All)
# ==========================================

# (1) 수치형 스케일링 (StandardScaler)
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train) # Train으로 fit!
X_num_val   = scaler.transform(X_num_val)       # Val은 transform만
X_num_test  = scaler.transform(X_num_test)      # Test도 transform만

# (2) 범주형 인코딩 (LabelEncoder)
label_encoders = {}
for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    # 주의: Train에 없는 카테고리가 Test에 있으면 에러납니다.
    # 만약 그런 경우가 있다면 전체 데이터를 합쳐서 fit 하거나 예외처리가 필요합니다.
    le.fit(X_cat_train[:, i]) 
    
    X_cat_train[:, i] = le.transform(X_cat_train[:, i])
    X_cat_val[:, i]   = le.transform(X_cat_val[:, i])
    X_cat_test[:, i]  = le.transform(X_cat_test[:, i])
    
    label_encoders[col] = le
    
X_num_train, X_cat_train, y_train = to_tensor(X_num_train, X_cat_train, y_train)
X_num_val,   X_cat_val,   y_val   = to_tensor(X_num_val, X_cat_val, y_val)
X_num_test,  X_cat_test,  y_test  = to_tensor(X_num_test, X_cat_test, y_test)

# DataLoader 생성
batch_size = 64
train_dl = DataLoader(TensorDataset(X_num_train, X_cat_train, y_train), batch_size=batch_size, shuffle=True)
val_dl   = DataLoader(TensorDataset(X_num_val, X_cat_val, y_val), batch_size=batch_size, shuffle=False)
test_dl  = DataLoader(TensorDataset(X_num_test, X_cat_test, y_test), batch_size=batch_size, shuffle=False)

# ==========================================
# 4. 모델 정의 및 학습
# ==========================================
cardinalities = [len(le.classes_) for le in label_encoders.values()]

model = rtdl.FTTransformer.make_default(
    n_num_features=X_num_train.shape[1],
    cat_cardinalities=cardinalities,
    last_layer_query_idx=[-1],
    d_out=2
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = FocalLoss(gamma=2.0, alpha=1.0).to(device)

print(f"Start Training on {device}...")

import copy

# ==========================================
# Early Stopping 설정
# ==========================================
MAX_EPOCHS = 100       # 최대 에폭 수
PATIENCE = 10          # 성능 향상이 없어도 참을 횟수
best_val_loss = float('inf') # 가장 낮은 Val Loss를 기록할 변수
best_val_auroc = float('-inf')  # <-- 추가
patience_counter = 0   # 현재 몇 번 참았는지 카운트
best_model_state = None # 베스트 모델 가중치 저장용

print(f"Start Training on {device} (Max Epochs: {MAX_EPOCHS}, Patience: {PATIENCE})...")

for epoch in range(MAX_EPOCHS):
    # --- [Training] ---
    model.train()
    train_loss = 0
    for x_n, x_c, label in train_dl:
        x_n, x_c, label = x_n.to(device), x_c.to(device), label.to(device)
        
        optimizer.zero_grad()
        logits = model(x_n, x_c)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_dl)

    # --- [Validation] ---
    # 아까 만든 평가 함수 사용
    val_metrics = evaluate_model(model, val_dl, criterion, device)
    current_val_auroc = val_metrics['auroc']
    
    print(f"Epoch {epoch+1:03d} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_metrics['loss']:.4f} | "
          f"AUROC: {val_metrics['auroc']:.4f} | "
          f"AUPRC: {val_metrics['auprc']:.4f}")

    # ==========================================
    # [핵심] Early Stopping 로직
    # ==========================================
    if current_val_auroc > best_val_auroc:
        # 1. 성능이 갱신됨 -> 저장하고 카운터 초기화
        best_val_auroc = current_val_auroc
        best_model_state = copy.deepcopy(model.state_dict()) # 현재 모델 복사
        patience_counter = 0
        # (선택) 파일로도 저장하고 싶으면:
        # torch.save(model.state_dict(), 'best_model.pth')
        print(f"    -> Best Model Saved! (Val Loss: {best_val_auroc:.4f})")
        
    else:
        # 2. 성능이 안 좋아짐 -> 카운터 증가
        patience_counter += 1
        print(f"    -> No improvement ({patience_counter}/{PATIENCE})")
        
        if patience_counter >= PATIENCE:
            print(f"\n[Early Stopping] 학습 종료! {PATIENCE}회 연속 성능 향상 없음.")
            break

# ==========================================
# 학습 종료 후, 가장 좋았던 모델로 복구
# ==========================================
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("\nBest Model weights loaded for Final Testing.")

# --- [Final Test] ---
test_metrics = evaluate_model(model, test_dl, criterion, device)

print(f"Final Test AUROC: {test_metrics['auroc']:.4f}")
print(f"AUROC: {test_metrics['auroc']:.4f} | "
      f"AUPRC: {test_metrics['auprc']:.4f}"
      )

import pandas as pd
import torch.nn.functional as F

# 1. 모델을 평가 모드로 전환
model.eval()

# 2. Test Set 전체에 대한 예측값(확률) 추출
# (DataLoader 순서가 섞이지 않도록 shuffle=False 여야 합니다 -> 아까 코드에선 False였으니 OK)
all_probs = []
all_preds = []
all_targets = []

with torch.no_grad():
    for x_n, x_c, y in test_dl:
        x_n, x_c, y = x_n.to(device), x_c.to(device), y.to(device)
        logits = model(x_n, x_c)
        
        # 사망(1)일 확률
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

# 3. 데이터프레임 만들기 (분석하기 쉽게)
# test_df는 아까 csv에서 읽어온 원본 데이터프레임입니다.
results_df = test_df.copy()

# 여기에 예측 결과 추가
results_df['prob'] = all_probs
results_df['pred'] = all_preds
results_df['target'] = all_targets # 확실하게 하기 위해 y값도 다시 넣음

# 확인: 민감 변수(ETHNICITY)와 예측값이 한 표에 있는지
print(results_df[['ETHNICITY', 'target', 'pred', 'prob']].head())

from evalutate import check_fairness
print("\n=== Fairness Audit Report ===")

# 함수 호출 (컬럼명은 본인 데이터에 맞게 수정)
report_df = check_fairness(
    df=results_df, 
    sensitive_col='ETHNICITY', 
    target_col='hospital_mortality', 
    prob_col='prob', 
    pred_col='pred'
)

# 결과 출력 (모든 컬럼 보기)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(report_df)

# (선택) 결과를 파일로 저장
report_df.to_csv("fairness_report.csv")
print("\n리포트가 'fairness_report.csv'로 저장되었습니다.")

# === ONNX export ===
model.eval()
model_cpu = model.to("cpu")

# 더미 입력 (배치 1)
dummy_x_num = torch.zeros(1, X_num_train.shape[1], dtype=torch.float32)
dummy_x_cat = torch.zeros(1, X_cat_train.shape[1], dtype=torch.long)

torch.onnx.export(
    model_cpu,
    (dummy_x_num, dummy_x_cat),
    "model_viz.onnx",
    input_names=["x_num", "x_cat"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "x_num": {0: "batch"},
        "x_cat": {0: "batch"},
        "output": {0: "batch"},
    },
)
print("model_viz.onnx 저장 완료! https://netron.app 에서 열어보세요.")
