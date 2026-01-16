import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from split_data.utils import load_data, merged_by_sub, drop_all_null_cols

# Load preprocessed data, and merged
df_x, df_a, df_y = load_data()
df = merged_by_sub(df_x, df_a)
df = merged_by_sub(df, df_y)

# NULL 값만 가지는 feature drop
df = drop_all_null_cols(df)

# 1. 복합 Stratify Key 생성
# 인종 + 보험 + 사망여부를 합쳐서 하나의 범주형 변수처럼 만듭니다.
df['stratify_key'] = (
    df['ETHNICITY'].astype(str) + "_" + 
    df['hospital_mortality'].astype(str)
)

# 2. StratifiedGroupKFold 설정
# 10개 셋으로 나눠서 7(Train) : 1.5(Val) : 1.5(Test) 비율을 맞추기 위한 준비
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

# 전체 데이터에서 Test 셋 15%(약 2/10)를 먼저 떼어냅니다.
# groups에는 SUBJECT_ID를, y에는 위에서 만든 복합 키를 넣습니다.
train_val_idx, test_idx = next(sgkf.split(df, df['stratify_key'], groups=df['SUBJECT_ID']))

# 나머지 85%를 다시 Train과 Val로 나눕니다.
train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# 다시 split (대략 8:2 비율로 나누면 전체 기준 약 7:1.5가 됨)
sgkf_sub = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf_sub.split(train_val_df, train_val_df['stratify_key'], groups=train_val_df['SUBJECT_ID']))

train_df = train_val_df.iloc[train_idx]
val_df = train_val_df.iloc[val_idx]

print(f"Train 환자 수: {train_df['SUBJECT_ID'].nunique()}, 사망률: {train_df['hospital_mortality'].mean():.2%}")
print(f"Val 환자 수: {val_df['SUBJECT_ID'].nunique()}, 사망률: {val_df['hospital_mortality'].mean():.2%}")
print(f"Test 환자 수: {test_df['SUBJECT_ID'].nunique()}, 사망률: {test_df['hospital_mortality'].mean():.2%}")

def save_and_clean(df, name, path):
    df = df.drop(columns=['stratify_key']) # 임시 키 삭제
    df.to_csv(os.path.join(path, f"{name}.csv"), index=False)

base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, '..', 'data', 'final_split')

os.makedirs(output_path, exist_ok=True)

save_and_clean(train_df, 'train', output_path)
save_and_clean(val_df, 'val', output_path)
save_and_clean(test_df, 'test', output_path)
