from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, confusion_matrix)
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # 평가 모드
    
    total_loss = 0
    all_targets = []
    all_probs = [] # 확률값 저장 (AUC 계산용)
    all_preds = [] # 0 or 1 예측값 저장 (Acc 계산용)
    
    with torch.no_grad():
        for x_num, x_cat, y in dataloader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            
            # 1. 모델 예측 (Logits)
            logits = model(x_num, x_cat)
            
            # 2. Loss 계산
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            # 3. 확률(Probability) 계산 [중요!]
            # Softmax를 거쳐야 0~1 사이 확률이 나옵니다.
            # [:, 1]은 'Class 1(사망/양성)'일 확률만 가져오는 것입니다.
            probs = F.softmax(logits, dim=1)[:, 1]
            
            # 4. 0 또는 1로 분류 (Threshold 0.5 기준)
            preds = logits.argmax(dim=1)
            
            # 5. 나중에 한꺼번에 계산하기 위해 리스트에 저장 (CPU로 이동)
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # 전체 데이터에 대한 지표 계산
    metrics = {
        'loss': total_loss / len(dataloader),
        'acc': accuracy_score(all_targets, all_preds),
        'auroc': roc_auc_score(all_targets, all_probs),
        'auprc': average_precision_score(all_targets, all_probs)
    }
    
    return metrics

def check_fairness(df, sensitive_col, target_col='target', prob_col='prob', pred_col='pred', min_samples=10):
    
    groups = df[sensitive_col].unique()
    report = []
    
    for group in groups:
        sub_df = df[df[sensitive_col] == group]
        
        if len(sub_df) < min_samples:
            continue
            
        # 1. 기본 메트릭
        acc = accuracy_score(sub_df[target_col], sub_df[pred_col])
        
        try:
            auroc = roc_auc_score(sub_df[target_col], sub_df[prob_col])
        except ValueError:
            auroc = None
            
        try:
            auprc = average_precision_score(sub_df[target_col], sub_df[prob_col])
        except ValueError:
            auprc = None

        # 2. Confusion Matrix 요소 추출 [핵심 수정]
        # labels=[0, 1]을 지정해야 데이터에 특정 클래스가 없어도 순서(TN, FP, FN, TP)가 고정됨
        cm = confusion_matrix(sub_df[target_col], sub_df[pred_col], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # 3. 파생 지표 계산
        selection_rate = sub_df[pred_col].mean()
        
        # ZeroDivisionError 방지
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall (Sensitivity)
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0 # 1 - Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # Precision (추가하면 좋음)
        
        report.append({
            'Group': group,
            'Count': len(sub_df),
            'AUROC': round(auroc, 4) if auroc is not None else None,
            'AUPRC': round(auprc, 4) if auprc is not None else None,
            'Acc': round(acc, 4),
            # --- [Confusion Matrix Raw Counts] ---
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            # -------------------------------------
            'Recall(TPR)': round(tpr, 4),
            'FPR': round(fpr, 4),
            'Precision': round(precision, 4),
            'Sel_Rate': round(selection_rate, 4)
        })
    
    result_df = pd.DataFrame(report)
    if not result_df.empty:
        result_df = result_df.set_index('Group')
        result_df = result_df.sort_values(by='Count', ascending=False)
        
    return result_df