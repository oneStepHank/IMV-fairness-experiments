import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import config

# --- 설정 ---
MIMIC_DIR = config.MIMIC_DIR
DATA_PATH = config.DATA_PATH
CHUNK_SIZE = config.CHUNK_SIZE

print("--- 0. 초기 설정 및 라이브러리 임포트 완료 ---")

# --- 헬퍼 함수 ---
def get_mimic_col(df, target_name):
    for col in df.columns:
        if col.upper() == target_name.upper():
            return col
    return None

# ==============================================================================
# 1. 초기 데이터 로드
# ==============================================================================
print("\n--- 1. 초기 데이터 로드 ---")
print("⏳ 필수 MIMIC-III 테이블 로드 중...")
df_a = pd.read_csv(MIMIC_DIR / 'ADMISSIONS.csv')
df_p = pd.read_csv(MIMIC_DIR / 'PATIENTS.csv')
df_i = pd.read_csv(MIMIC_DIR / 'ICUSTAYS.csv')
print("✅ 필수 테이블 로드 완료: ADMISSIONS, PATIENTS, ICUSTAYS")

# 질병 중증도 점수 로드
print("⏳ 질병 중증도 점수 파일 로드 중...")
score_files = {
    'OASIS': Path('./data/oasis.csv'),
    'SAPSII': Path('./data/sapsii.csv'),
    'SOFA': Path('./data/sofa.csv')
}
df_scores = {}
for score_name, score_path in score_files.items():
    if score_path.exists():
        df_scores[score_name] = pd.read_csv(score_path, low_memory=False)
        print(f"   ✅ {score_name} 로드 완료: {len(df_scores[score_name])} rows")
    else:
        print(f"   ⚠️ {score_name} 파일을 찾을 수 없습니다: {score_path}")
        df_scores[score_name] = None

# ==============================================================================
# 2. 코호트 선정 및 데이터 추출 (통합 프로세스)
# ==============================================================================
print("\n" + "="*80)
print("코호트 선정 및 데이터 추출 (통합 프로세스)")
print("="*80)

# 2.1. 성인 환자 필터링
print("⏳ 환자 및 입원 정보 처리 중...")
dob_col = get_mimic_col(df_p, 'DOB')
gender_col = get_mimic_col(df_p, 'GENDER')
df_p_select = df_p[['SUBJECT_ID', dob_col, gender_col]].copy()
df_p_select.rename(columns={dob_col: 'DOB', gender_col: 'GENDER'}, inplace=True)

df_i = pd.merge(df_i, df_p_select, on='SUBJECT_ID', how='left')
df_i['DOB'] = pd.to_datetime(df_i['DOB'], errors='coerce')
df_i['INTIME'] = pd.to_datetime(df_i['INTIME'], errors='coerce')
df_i.dropna(subset=['DOB', 'INTIME'], inplace=True)

df_i['ADMIT_YEAR'] = df_i['INTIME'].dt.year
df_i['BIRTH_YEAR'] = df_i['DOB'].dt.year
df_i['AGE'] = df_i['ADMIT_YEAR'] - df_i['BIRTH_YEAR']

# 성인 환자 (18세 이상 90세 미만)
# Note: MIMIC-III에서 90세 이상은 프라이버시를 위해 300세 이상으로 표시되므로
# 실제 나이 계산 시 매우 큰 값이 나올 수 있음. 이를 필터링으로 제외.
df_cohort = df_i[(df_i['AGE'] >= 18) & (df_i['AGE'] < 90)].copy()
n_adult = df_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 1: Adult Patients (18 years old and < 90 years old)")
print(f"   Our cohort:   {n_adult:,} patients")
print(f"   Paper:        38,597 patients")

# 2.2. MV 환자 식별 (ICD-9 + CHARTEVENTS 통합)
print("\n⏳ MV 환자 식별 및 활력징후 추출 (Single Pass)...")

# (1) ICD-9 기반 MV 환자 식별
df_proc = pd.read_csv(MIMIC_DIR / 'PROCEDURES_ICD.csv', low_memory=False)
proc_icd_col = get_mimic_col(df_proc, 'ICD9_CODE')
mv_codes = ['9670', '9671', '9672']
mv_hadm_ids_icd9 = set(df_proc[df_proc[proc_icd_col].astype(str).isin(mv_codes)]['HADM_ID'].unique())
print(f"   - ICD-9 코드로 식별된 MV 환자(HADM_ID): {len(mv_hadm_ids_icd9):,}명")

# (2) CHARTEVENTS 기반 MV 식별 및 활력징후 추출
# SQL Logic Implementation
# MechVent ITEMIDs from the provided SQL
MV_ITEMIDS_SQL = [
    720, 223849, 223848, 223849, 467, # Settings with value checks
    445, 448, 449, 450, 1340, 1486, 1600, 224687, # Minute volume
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686, # Tidal volume
    218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, # RespPressure
    221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, # Insp pressure
    543, # PlateauPressure
    5865, 5866, 224707, 224709, 224705, 224706, # APRV pressure
    60, 437, 505, 506, 686, 220339, 224700, # PEEP
    3459, # High pressure relief
    501, 502, 503, 224702, # PCV
    223, 667, 668, 669, 670, 671, 672, # TCPCV
    224701 # PSVlevel
]
MV_ITEMIDS_SET = set(MV_ITEMIDS_SQL)

# 활력징후 ITEMID
VITAL_SIGN_MAP = {
    'HR': [211, 220045], 
    'SBP': [51, 455, 220179, 220050], 
    'DBP': [8368, 8440, 220180, 220051],
    'MAP': [52, 456, 220181, 220052], 
    'Temp': [678, 679, 223761, 223762]
}
VITAL_SIGN_ITEMIDS = [item for sublist in VITAL_SIGN_MAP.values() for item in sublist]
VITAL_SIGN_ITEMIDS_SET = set(VITAL_SIGN_ITEMIDS)

# 통합 ITEMID 세트 (필터링용)
ALL_TARGET_ITEMIDS = MV_ITEMIDS_SET | VITAL_SIGN_ITEMIDS_SET

# 데이터 수집
mv_icustay_ids_chart = set()
ce_24hr_list = []

chartevents_path = MIMIC_DIR / 'CHARTEVENTS.csv'
file_size = os.path.getsize(chartevents_path)
estimated_chunks = max(1, file_size // (CHUNK_SIZE * 100))

reader_ce = pd.read_csv(
    chartevents_path, 
    chunksize=CHUNK_SIZE, 
    low_memory=False, 
    iterator=True
)

# 코호트의 ICUSTAY_ID 목록 (성인 환자 전체 대상)
adult_icustay_ids = set(df_cohort['ICUSTAY_ID'].unique())

print(f"⏳ Reading CHARTEVENTS.csv (Raw)... This may take 10-15 minutes.")
for chunk in tqdm(reader_ce, desc="Processing CHARTEVENTS", total=estimated_chunks):
    chunk.columns = [c.upper() for c in chunk.columns]
    
    # 1. 성인 환자 필터링
    chunk = chunk[chunk['ICUSTAY_ID'].isin(adult_icustay_ids)]
    if chunk.empty: continue
    
    # 2. 관심 ITEMID 필터링
    chunk = chunk[chunk['ITEMID'].isin(ALL_TARGET_ITEMIDS)]
    if chunk.empty: continue
    
    # 3. MV 환자 식별 (SQL Logic 적용)
    # SQL Logic:
    # when itemid = 720 and value != 'Other/Remarks' THEN 1
    # when itemid = 223848 and value != 'Other' THEN 1
    # when itemid = 223849 then 1
    # when itemid = 467 and value = 'Ventilator' THEN 1
    # else (other itemids) THEN 1
    
    mv_chunk = chunk[chunk['ITEMID'].isin(MV_ITEMIDS_SET)].copy()
    if not mv_chunk.empty:
        if 'VALUE' in mv_chunk.columns:
            mv_chunk['VALUE'] = mv_chunk['VALUE'].astype(str)
            
            # 조건별 마스크 생성
            mask_720 = (mv_chunk['ITEMID'] == 720) & (mv_chunk['VALUE'] != 'Other/Remarks')
            mask_223848 = (mv_chunk['ITEMID'] == 223848) & (mv_chunk['VALUE'] != 'Other')
            mask_467 = (mv_chunk['ITEMID'] == 467) & (mv_chunk['VALUE'] == 'Ventilator')
            
            # 나머지 ITEMID는 존재하기만 하면 MV로 간주
            other_mv_itemids = MV_ITEMIDS_SET - {720, 223848, 467}
            mask_others = mv_chunk['ITEMID'].isin(other_mv_itemids)
            
            # 최종 MV 마스크
            final_mv_mask = mask_720 | mask_223848 | mask_467 | mask_others
            
            valid_mv = mv_chunk[final_mv_mask]
            mv_icustay_ids_chart.update(valid_mv['ICUSTAY_ID'].unique())
        else:
            # VALUE 컬럼이 없으면 ITEMID만으로 판단 (예외적 상황)
            mv_icustay_ids_chart.update(mv_chunk['ICUSTAY_ID'].unique())
            
    # 4. 활력징후 데이터 수집 (일단 저장, 나중에 MV 환자만 필터링)
    vital_chunk = chunk[chunk['ITEMID'].isin(VITAL_SIGN_ITEMIDS_SET)]
    if not vital_chunk.empty:
        ce_24hr_list.append(vital_chunk[['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']])

print(f"   - CHARTEVENTS로 식별된 MV 환자(ICUSTAY_ID): {len(mv_icustay_ids_chart):,}명")

# (3) MV 환자 통합 (ICD-9 OR CHARTEVENTS)
# ICD-9 HADM_ID -> ICUSTAY_ID 변환
mv_icustay_ids_icd9 = set(df_cohort[df_cohort['HADM_ID'].isin(mv_hadm_ids_icd9)]['ICUSTAY_ID'].unique())
final_mv_icustay_ids = mv_icustay_ids_icd9 | mv_icustay_ids_chart

# 코호트 필터링
df_mv_cohort = df_cohort[df_cohort['ICUSTAY_ID'].isin(final_mv_icustay_ids)].copy()
n_mv_combined = df_mv_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 2: Mechanically Ventilated Patients (Combined Filter)")
print(f"   Our cohort:   {n_mv_combined:,} patients")
print(f"   Paper:        28,530 patients")

# (4) 활력징후 데이터 정리 (MV 환자 & 24시간 이내)
print("\n⏳ 활력징후 데이터 정리 중...")
if ce_24hr_list:
    df_ce_all = pd.concat(ce_24hr_list)
    
    # MV 환자만 남기기
    df_ce_all = df_ce_all[df_ce_all['ICUSTAY_ID'].isin(final_mv_icustay_ids)]
    
    # 시간 필터링
    cohort_times = df_mv_cohort[['ICUSTAY_ID', 'INTIME']].drop_duplicates()
    df_ce_all['CHARTTIME'] = pd.to_datetime(df_ce_all['CHARTTIME'], errors='coerce')
    df_ce_all = pd.merge(df_ce_all, cohort_times, on='ICUSTAY_ID', how='inner')
    
    df_ce_24hr = df_ce_all[
        (df_ce_all['CHARTTIME'] >= df_ce_all['INTIME']) & 
        (df_ce_all['CHARTTIME'] <= df_ce_all['INTIME'] + pd.Timedelta(hours=24))
    ][['ICUSTAY_ID', 'ITEMID', 'VALUENUM']]
    
    print(f"   ✅ 활력징후 데이터 정리 완료: {len(df_ce_24hr):,} rows")
else:
    df_ce_24hr = pd.DataFrame(columns=['ICUSTAY_ID', 'ITEMID', 'VALUENUM'])
    print("   ⚠️ 활력징후 데이터가 없습니다.")

# 2.3. First ICU Stay 필터링
df_mv_cohort.sort_values(by=['SUBJECT_ID', 'INTIME'], inplace=True)
df_mv_cohort = df_mv_cohort.drop_duplicates(subset=['SUBJECT_ID'], keep='first').copy()
n_first_icu = df_mv_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 3: First ICU Stay Only")
print(f"   Our cohort:   {n_first_icu:,} patients")
print(f"   Paper:        28,530 patients (selection criteria met)")
print(f"   Difference:   {n_first_icu - 28530:+,} ({(n_first_icu/28530-1)*100:+.1f}%)")

# Step 4: LOS < 24시간 제외
if 'LOS' in df_mv_cohort.columns:
    n_before_los = df_mv_cohort['SUBJECT_ID'].nunique()
    df_mv_cohort = df_mv_cohort[df_mv_cohort['LOS'] >= 1.0].copy()
    n_after_los = df_mv_cohort['SUBJECT_ID'].nunique()
    print(f"\n📊 Step 4: LOS ≥ 24 hours")
    print(f"   Excluded:     {n_before_los - n_after_los:,} patients")
    print(f"   Remaining:    {n_after_los:,} patients")

# Step 5: 병원 사망률 정보 추가 및 최종 코호트
hosp_expire_col = get_mimic_col(df_a, 'HOSPITAL_EXPIRE_FLAG')
cols_to_merge = ['HADM_ID', hosp_expire_col, 'INSURANCE', 'ETHNICITY']
df_mv_cohort = pd.merge(df_mv_cohort, df_a[cols_to_merge], on='HADM_ID', how='left')
df_mv_cohort['hospital_mortality'] = df_mv_cohort[hosp_expire_col]

n_before_mortality = df_mv_cohort['SUBJECT_ID'].nunique()
df_mv_cohort.dropna(subset=['hospital_mortality'], inplace=True)
n_final = df_mv_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 5: Final Cohort (after logic check)")
print(f"   Excluded:     {n_before_mortality - n_final:,} patients (missing mortality data)")
print(f"   Our cohort:   {n_final:,} patients")
print(f"   Paper:        25,659 patients")
print(f"   Difference:   {n_final - 25659:+,} ({(n_final/25659-1)*100:+.1f}%)")

# 사망률 통계
n_survivors = (df_mv_cohort['hospital_mortality'] == 0).sum()
n_deaths = (df_mv_cohort['hospital_mortality'] == 1).sum()
mortality_rate = n_deaths / n_final * 100

print(f"\n📊 Mortality Statistics")
print(f"   Survivors:    {n_survivors:,} ({n_survivors/n_final*100:.1f}%)")
print(f"   Deaths:       {n_deaths:,} ({mortality_rate:.1f}%)")
print(f"   Paper:        13,987 survivors (54.5%), 11,672 deaths (45.5%)")

print("="*80)

# ETHNICITY 매핑
print("\n-> ETHNICITY 매핑 중...")
import json

ethnicity_map_file = Path('ethnicity.json')
if ethnicity_map_file.exists():
    with open(ethnicity_map_file, 'r', encoding='utf-8') as f:
        ethnicity_config = json.load(f)
    
    ethnicity_mapping = {}
    for mapping in ethnicity_config['mappings']:
        main_code = mapping['code']
        for detail_code in mapping['lists']:
            ethnicity_mapping[detail_code.upper()] = main_code
    
    df_mv_cohort['ETHNICITY'] = df_mv_cohort['ETHNICITY'].str.upper().map(ethnicity_mapping).fillna('OTHER')
    
    print(f"   ✅ ETHNICITY 매핑 완료")
    print(f"   매핑된 그룹: {df_mv_cohort['ETHNICITY'].value_counts().to_dict()}")
else:
    print(f"   ⚠️ ethnicity.json 파일을 찾을 수 없습니다. 매핑을 건너뜁니다.")


# 2.2. MV 환자 식별 (ICD-9 + CHARTEVENTS 통합)
print("\n⏳ MV 환자 식별 및 활력징후 추출 (Single Pass)...")

# (1) ICD-9 기반 MV 환자 식별
df_proc = pd.read_csv(MIMIC_DIR / 'PROCEDURES_ICD.csv', low_memory=False)
proc_icd_col = get_mimic_col(df_proc, 'ICD9_CODE')
mv_codes = ['9670', '9671', '9672']
mv_hadm_ids_icd9 = set(df_proc[df_proc[proc_icd_col].astype(str).isin(mv_codes)]['HADM_ID'].unique())
print(f"   - ICD-9 코드로 식별된 MV 환자(HADM_ID): {len(mv_hadm_ids_icd9):,}명")

# (2) CHARTEVENTS 기반 MV 식별 및 활력징후 추출
# MV 관련 ITEMID (SQL 로직 기반)
MV_ITEMIDS = [
    720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687,
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
    218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747,
    221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
    543, 5865, 5866, 224707, 224709, 224705, 224706,
    60, 437, 505, 506, 686, 220339, 224700, 3459, 501, 502, 503, 224702,
    223, 667, 668, 669, 670, 671, 672, 224701
]
MV_ITEMIDS_SET = set(MV_ITEMIDS)

# 활력징후 ITEMID
VITAL_SIGN_MAP = {
    'HR': [211, 220045], 
    'SBP': [51, 455, 220179, 220050], 
    'DBP': [8368, 8440, 220180, 220051],
    'MAP': [52, 456, 220181, 220052], 
    'Temp': [678, 679, 223761, 223762]
}
VITAL_SIGN_ITEMIDS = [item for sublist in VITAL_SIGN_MAP.values() for item in sublist]
VITAL_SIGN_ITEMIDS_SET = set(VITAL_SIGN_ITEMIDS)

# 통합 ITEMID 세트 (필터링용)
ALL_TARGET_ITEMIDS = MV_ITEMIDS_SET | VITAL_SIGN_ITEMIDS_SET

# 데이터 수집
mv_icustay_ids_chart = set()
ce_24hr_list = []

chartevents_path = MIMIC_DIR / 'CHARTEVENTS.csv'
file_size = os.path.getsize(chartevents_path)
estimated_chunks = max(1, file_size // (CHUNK_SIZE * 100))

reader_ce = pd.read_csv(
    chartevents_path, 
    chunksize=CHUNK_SIZE, 
    low_memory=False, 
    iterator=True
)

# 코호트의 ICUSTAY_ID 목록 (성인 환자 전체 대상)
adult_icustay_ids = set(df_cohort['ICUSTAY_ID'].unique())

print(f"⏳ Reading CHARTEVENTS.csv (Raw)... This may take 10-15 minutes.")
for chunk in tqdm(reader_ce, desc="Processing CHARTEVENTS", total=estimated_chunks):
    chunk.columns = [c.upper() for c in chunk.columns]
    
    # 1. 성인 환자 필터링
    chunk = chunk[chunk['ICUSTAY_ID'].isin(adult_icustay_ids)]
    if chunk.empty: continue
    
    # 2. 관심 ITEMID 필터링
    chunk = chunk[chunk['ITEMID'].isin(ALL_TARGET_ITEMIDS)]
    if chunk.empty: continue
    
    # 3. MV 환자 식별 (CHARTEVENTS 기반)
    mv_chunk = chunk[chunk['ITEMID'].isin(MV_ITEMIDS_SET)]
    if not mv_chunk.empty:
        # VALUE 필터링 (간소화: 'Other' 등 제외)
        if 'VALUE' in mv_chunk.columns:
            val_str = mv_chunk['VALUE'].astype(str).str.lower()
            # ITEMID 720, 223848의 'other' 제외
            exclude_mask = (mv_chunk['ITEMID'].isin([720, 223848])) & (val_str.str.contains('other', na=False))
            
            valid_mv = mv_chunk[~exclude_mask]
            mv_icustay_ids_chart.update(valid_mv['ICUSTAY_ID'].unique())
        else:
            mv_icustay_ids_chart.update(mv_chunk['ICUSTAY_ID'].unique())
            
    # 4. 활력징후 데이터 수집 (일단 저장, 나중에 MV 환자만 필터링)
    vital_chunk = chunk[chunk['ITEMID'].isin(VITAL_SIGN_ITEMIDS_SET)]
    if not vital_chunk.empty:
        ce_24hr_list.append(vital_chunk[['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']])

print(f"   - CHARTEVENTS로 식별된 MV 환자(ICUSTAY_ID): {len(mv_icustay_ids_chart):,}명")

# (3) MV 환자 통합 (ICD-9 OR CHARTEVENTS)
# ICD-9 HADM_ID -> ICUSTAY_ID 변환
mv_icustay_ids_icd9 = set(df_cohort[df_cohort['HADM_ID'].isin(mv_hadm_ids_icd9)]['ICUSTAY_ID'].unique())
final_mv_icustay_ids = mv_icustay_ids_icd9 | mv_icustay_ids_chart

# 코호트 필터링
df_mv_cohort = df_cohort[df_cohort['ICUSTAY_ID'].isin(final_mv_icustay_ids)].copy()
n_mv_combined = df_mv_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 2: Mechanically Ventilated Patients (Combined Filter)")
print(f"   Our cohort:   {n_mv_combined:,} patients")
print(f"   Paper:        28,530 patients")

# (4) 활력징후 데이터 정리 (MV 환자 & 24시간 이내)
print("\n⏳ 활력징후 데이터 정리 중...")
if ce_24hr_list:
    df_ce_all = pd.concat(ce_24hr_list)
    
    # MV 환자만 남기기
    df_ce_all = df_ce_all[df_ce_all['ICUSTAY_ID'].isin(final_mv_icustay_ids)]
    
    # 시간 필터링
    cohort_times = df_mv_cohort[['ICUSTAY_ID', 'INTIME']].drop_duplicates()
    df_ce_all['CHARTTIME'] = pd.to_datetime(df_ce_all['CHARTTIME'], errors='coerce')
    df_ce_all = pd.merge(df_ce_all, cohort_times, on='ICUSTAY_ID', how='inner')
    
    df_ce_24hr = df_ce_all[
        (df_ce_all['CHARTTIME'] >= df_ce_all['INTIME']) & 
        (df_ce_all['CHARTTIME'] <= df_ce_all['INTIME'] + pd.Timedelta(hours=24))
    ][['ICUSTAY_ID', 'ITEMID', 'VALUENUM']]
    
    print(f"   ✅ 활력징후 데이터 정리 완료: {len(df_ce_24hr):,} rows")
else:
    df_ce_24hr = pd.DataFrame(columns=['ICUSTAY_ID', 'ITEMID', 'VALUENUM'])
    print("   ⚠️ 활력징후 데이터가 없습니다.")

# 2.3. First ICU Stay 필터링
df_mv_cohort.sort_values(by=['SUBJECT_ID', 'INTIME'], inplace=True)
df_mv_cohort = df_mv_cohort.drop_duplicates(subset=['SUBJECT_ID'], keep='first').copy()
n_first_icu = df_mv_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 3: First ICU Stay Only")
print(f"   Our cohort:   {n_first_icu:,} patients")
print(f"   Paper:        28,530 patients (selection criteria met)")
print(f"   Difference:   {n_first_icu - 28530:+,} ({(n_first_icu/28530-1)*100:+.1f}%)")

# Step 4: LOS < 24시간 제외
if 'LOS' in df_mv_cohort.columns:
    n_before_los = df_mv_cohort['SUBJECT_ID'].nunique()
    df_mv_cohort = df_mv_cohort[df_mv_cohort['LOS'] >= 1.0].copy()
    n_after_los = df_mv_cohort['SUBJECT_ID'].nunique()
    print(f"\n📊 Step 4: LOS ≥ 24 hours")
    print(f"   Excluded:     {n_before_los - n_after_los:,} patients")
    print(f"   Remaining:    {n_after_los:,} patients")

# Step 5: 병원 사망률 정보 추가 및 최종 코호트
hosp_expire_col = get_mimic_col(df_a, 'HOSPITAL_EXPIRE_FLAG')
cols_to_merge = ['HADM_ID', hosp_expire_col, 'INSURANCE', 'ETHNICITY']
df_mv_cohort = pd.merge(df_mv_cohort, df_a[cols_to_merge], on='HADM_ID', how='left')
df_mv_cohort['hospital_mortality'] = df_mv_cohort[hosp_expire_col]

n_before_mortality = df_mv_cohort['SUBJECT_ID'].nunique()
df_mv_cohort.dropna(subset=['hospital_mortality'], inplace=True)
n_final = df_mv_cohort['SUBJECT_ID'].nunique()

print(f"\n📊 Step 5: Final Cohort (after logic check)")
print(f"   Excluded:     {n_before_mortality - n_final:,} patients (missing mortality data)")
print(f"   Our cohort:   {n_final:,} patients")
print(f"   Paper:        25,659 patients")
print(f"   Difference:   {n_final - 25659:+,} ({(n_final/25659-1)*100:+.1f}%)")

# 사망률 통계
n_survivors = (df_mv_cohort['hospital_mortality'] == 0).sum()
n_deaths = (df_mv_cohort['hospital_mortality'] == 1).sum()
mortality_rate = n_deaths / n_final * 100

print(f"\n📊 Mortality Statistics")
print(f"   Survivors:    {n_survivors:,} ({n_survivors/n_final*100:.1f}%)")
print(f"   Deaths:       {n_deaths:,} ({mortality_rate:.1f}%)")
print(f"   Paper:        13,987 survivors (54.5%), 11,672 deaths (45.5%)")

print("="*80)

# ETHNICITY 매핑
print("\n-> ETHNICITY 매핑 중...")
import json

ethnicity_map_file = Path('ethnicity.json')
if ethnicity_map_file.exists():
    with open(ethnicity_map_file, 'r', encoding='utf-8') as f:
        ethnicity_config = json.load(f)
    
    ethnicity_mapping = {}
    for mapping in ethnicity_config['mappings']:
        main_code = mapping['code']
        for detail_code in mapping['lists']:
            ethnicity_mapping[detail_code.upper()] = main_code
    
    df_mv_cohort['ETHNICITY'] = df_mv_cohort['ETHNICITY'].str.upper().map(ethnicity_mapping).fillna('OTHER')
    
    print(f"   ✅ ETHNICITY 매핑 완료")
    print(f"   매핑된 그룹: {df_mv_cohort['ETHNICITY'].value_counts().to_dict()}")
else:
    print(f"   ⚠️ ethnicity.json 파일을 찾을 수 없습니다. 매핑을 건너뜁니다.")


# ==============================================================================
# 3. 특징 추출
# ==============================================================================
print("\n--- 3. 특징 추출 및 정리 ---")

# 3.1. DIAGNOSES_ICD 로드 (dtype=str 필수)
print("⏳ DIAGNOSES_ICD.csv 로드 중...")
df_diag = pd.read_csv(MIMIC_DIR / 'DIAGNOSES_ICD.csv', dtype={'ICD9_CODE': str}, low_memory=False)

# 3.2. LABEVENTS 로드 (Chunk 단위 처리)
print("⏳ LABEVENTS.csv 처리 중 (Chunk 단위)...")
LAB_COLS = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']
LAB_ITEMIDS = {
    'lactate': [818, 1531, 225668], 
    'hgb': [50811, 51006, 51222, 51634, 52028], 
    'bun': [50882, 51006, 50931], 
    'creatinine': [50912], 
    'wbc': [51300, 51301], 
    'glucose': [50809, 50938, 51240], 
    'ph': [50820, 50931]
}
ALL_LAB_ITEMIDS = [item for sublist in LAB_ITEMIDS.values() for item in sublist]
ALL_LAB_ITEMIDS_SET = set(ALL_LAB_ITEMIDS)

lab_data_list = []
labevents_path = MIMIC_DIR / 'LABEVENTS.csv'
lab_file_size = os.path.getsize(labevents_path)
lab_estimated_chunks = max(1, lab_file_size // (CHUNK_SIZE * 100))

# 코호트의 HADM_ID 목록
target_hadm_ids = set(df_mv_cohort['HADM_ID'].unique())

reader_lab = pd.read_csv(
    labevents_path, 
    usecols=LAB_COLS, 
    parse_dates=['CHARTTIME'], 
    chunksize=CHUNK_SIZE, 
    low_memory=False
)

for chunk in tqdm(reader_lab, desc="Processing LABEVENTS", total=lab_estimated_chunks):
    # 1. 코호트 환자 필터링 (HADM_ID 기준)
    chunk = chunk[chunk['HADM_ID'].isin(target_hadm_ids)]
    if chunk.empty: continue
    
    # 2. 관심 ITEMID 필터링
    chunk = chunk[chunk['ITEMID'].isin(ALL_LAB_ITEMIDS_SET)]
    if not chunk.empty:
        lab_data_list.append(chunk)

if lab_data_list:
    df_lab_filtered = pd.concat(lab_data_list)
    print(f"   ✅ LABEVENTS 데이터 로드 완료: {len(df_lab_filtered):,} rows")
else:
    df_lab_filtered = pd.DataFrame(columns=LAB_COLS)
    print("   ⚠️ LABEVENTS 데이터가 없습니다.")

# --- ICD-9 Comorbidity Mapping ---
# Based on: Quan et al. (2005) & MIMIC-Code
ICD9_MAP = {
    # Hypertension
    'Hypertension_uncomplicated': ['401'],
    'Hypertension_complicated': ['402', '403', '404', '405'],
    # Diabetes
    'Diabetes_uncomplicated': ['2500', '2501', '2502', '2503'],
    'Diabetes_complicated': ['2504', '2505', '2506', '2507', '2508', '2509'],
    # Others
    'Malignancy': [str(x) for x in range(140, 209)] + ['2386'],
    'Hematologic_disease': ['200', '201', '202', '203', '204', '205', '206', '207', '208'],
    'Metastasis': ['196', '197', '198', '199'],
    'Peripheral_vascular_disease': ['440', '441', '442', '443', '444', '447', '557', 'V434'],
    'Hypothyroidism': ['243', '244'],
    'Chronic_heart_failure': ['428'],
    'Stroke': ['430', '431', '432', '433', '434', '435', '436', '437', '438'],
    'Liver_disease': ['571', '570', '572'],
    # Angus Criteria
    'Sepsis': ['038', '99591', '99592', '78552'],
    'Respiratory_dysfunction': ['486', '51881', '51882', '51885', '78609'],
    'Cardiovascular_dysfunction': ['4580', '4588', '4589', '7855', '78551', '78559'],
    'Renal_dysfunction': ['580', '584', '585'],
    'Hepatic_dysfunction': ['570', '5722', '5733'],
    'Hematologic_dysfunction': ['2866', '2869', '2873', '2874', '2875'],
    'Metabolic_dysfunction': ['2762'],
    'Neurologic_dysfunction': ['293', '3481', '3483', '78001', '78009']
}

# 3.3. LABEVENTS 24시간 데이터 필터링 및 집계
print("-> LABEVENTS 24시간 데이터 필터링 및 집계...")
cohort_times_lab = df_mv_cohort[['HADM_ID', 'ICUSTAY_ID', 'INTIME']].drop_duplicates()
df_lab_merged = pd.merge(df_lab_filtered, cohort_times_lab, on='HADM_ID', how='inner')

valid_lab = df_lab_merged[
    (df_lab_merged['CHARTTIME'] >= df_lab_merged['INTIME']) & 
    (df_lab_merged['CHARTTIME'] <= df_lab_merged['INTIME'] + pd.Timedelta(hours=24))
]

# df_mv_cohort_final 초기화
df_mv_cohort_final = df_mv_cohort.copy()

# Lab 집계 (Min, Max, Mean)
for lab_name, itemids in LAB_ITEMIDS.items():
    df_item = valid_lab[valid_lab['ITEMID'].isin(itemids)]
    
    # 기본 프레임 (모든 환자 포함)
    df_agg = pd.DataFrame({'ICUSTAY_ID': df_mv_cohort_final['ICUSTAY_ID'].unique()})
    
    if not df_item.empty:
        df_stats = df_item.groupby('ICUSTAY_ID')['VALUENUM'].agg(['min', 'max', 'mean']).reset_index()
        df_stats.columns = ['ICUSTAY_ID', f'min_{lab_name}', f'max_{lab_name}', f'mean_{lab_name}']
        df_agg = pd.merge(df_agg, df_stats, on='ICUSTAY_ID', how='left')
    else:
        print(f"   ⚠️ {lab_name} 데이터가 없습니다. (NaN으로 채움)")
        for stat in ['min', 'max', 'mean']:
            df_agg[f'{stat}_{lab_name}'] = np.nan
            
    df_mv_cohort_final = pd.merge(df_mv_cohort_final, df_agg, on='ICUSTAY_ID', how='left')

# 3.4. 질병 중증도 점수 통합
print("-> 질병 중증도 점수 통합 중...")
for score_name, df_score in df_scores.items():
    if df_score is not None:
        score_col = score_name
        cols_to_merge = ['ICUSTAY_ID', score_col]
        
        if score_name == 'SOFA':
            sofa_subs = ['SOFA_Respiration', 'SOFA_Coagulation', 'SOFA_Liver', 
                         'SOFA_Cardiovascular', 'SOFA_CNS', 'SOFA_Renal']
            existing_subs = [col for col in df_score.columns if col in sofa_subs] # Check if sub-score columns exist
            cols_to_merge.extend(existing_subs)
            if existing_subs:
                print(f"      -> SOFA Sub-scores 병합: {existing_subs}")
            else:
                print("      ⚠️ SOFA Sub-scores 컬럼이 없습니다. (SOFA_Respiration 등)")

        if score_col in df_score.columns:
            df_mv_cohort_final = pd.merge(df_mv_cohort_final, df_score[cols_to_merge], on='ICUSTAY_ID', how='left')
            print(f"   ✅ {score_name} 점수 병합 완료 (결측: {df_mv_cohort_final[score_col].isnull().sum()})")
        else:
            print(f"   ⚠️ {score_name} 파일에 '{score_col}' 컬럼이 없습니다.")
            df_mv_cohort_final[score_name] = np.nan
            if score_name == 'SOFA':
                for sub in sofa_subs: df_mv_cohort_final[sub] = np.nan
    else:
        print(f"   ⚠️ {score_name} 데이터가 없어 NaN으로 설정합니다.")
        df_mv_cohort_final[score_name] = np.nan

# 3.5. 동반 질환 (Comorbidities) 추가
print("-> 동반 질환(Comorbidities) 변수 추가 중...")
diag_icd_col = get_mimic_col(df_diag, 'ICD9_CODE')
if diag_icd_col:
    # 문자열 매칭을 위해 컬럼 타입 확인
    df_diag[diag_icd_col] = df_diag[diag_icd_col].astype(str)
    
    for disease, codes in ICD9_MAP.items():
        # startswith 매칭 지원 (예: '401'은 '4010', '4019' 등을 포함해야 함)
        # MIMIC 코드는 소수점이 없음. Quan 코드는 3~4자리.
        # 정확한 매칭 + startswith 매칭 혼용 필요. 여기서는 prefix 매칭 사용.
        
        # 해당 코드로 시작하는 모든 진단 코드 찾기
        matched_codes = set()
        for code in codes:
            matched = df_diag[df_diag[diag_icd_col].str.startswith(code, na=False)][diag_icd_col].unique()
            matched_codes.update(matched)
        
        target_hadm_ids = df_diag[df_diag[diag_icd_col].isin(matched_codes)]['HADM_ID'].unique()
        df_mv_cohort_final[disease] = df_mv_cohort_final['HADM_ID'].isin(target_hadm_ids).astype(int)
        print(f"   - {disease}: {df_mv_cohort_final[disease].sum()}명")
else:
    print("   ⚠️ DIAGNOSES_ICD 테이블에서 ICD9_CODE 컬럼을 찾을 수 없습니다.")

# 3.6. 활력징후 (Vital Signs) 집계
print("-> 활력징후(Vital Signs) 집계 중...")
if not df_ce_24hr.empty:
    for vital_name, itemids in VITAL_SIGN_MAP.items():
        df_item = df_ce_24hr[df_ce_24hr['ITEMID'].isin(itemids)]
        if not df_item.empty:
            df_agg = df_item.groupby('ICUSTAY_ID')['VALUENUM'].agg(['min', 'max', 'mean']).reset_index()
            df_agg.columns = ['ICUSTAY_ID', f'min_{vital_name.lower()}', f'max_{vital_name.lower()}', f'mean_{vital_name.lower()}']
            df_mv_cohort_final = pd.merge(df_mv_cohort_final, df_agg, on='ICUSTAY_ID', how='left')
        else:
            for stat in ['min', 'max', 'mean']:
                df_mv_cohort_final[f'{stat}_{vital_name.lower()}'] = np.nan
else:
    print("   ⚠️ 활력징후 데이터가 없습니다.")

# ==============================================================================
# 4. 데이터 정리 및 저장
# ==============================================================================
print("\n--- 4. 최종 데이터 정리 및 저장 ---")

# 메타데이터 및 불필요 컬럼 제거 (모델 입력 변수만 남기기 위해)
# 유지해야 할 식별자 및 타겟: 'HADM_ID' (나중에 제거), 'hospital_mortality', 'ETHNICITY', 'INSURANCE'
meta_cols = [
    'ROW_ID', 'ICUSTAY_ID', 'HADM_ID',
    'INTIME', 'OUTTIME', 'ADMITTIME', 'DOB', 'DOD', 
    'ADMIT_YEAR', 'BIRTH_YEAR', 'ROW_ID_x', 'ROW_ID_y',
    'GENDER_x', 'GENDER_y', 'DOB_x', 'DOB_y',
    'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID',
    'LOS', 'DBSOURCE', 'HOSPITAL_EXPIRE_FLAG'
]
# 실제 존재하는 컬럼만 제거 (SUBJECT_ID는 유지)
existing_cols_to_drop = [c for c in meta_cols if c in df_mv_cohort_final.columns]
df_features = df_mv_cohort_final.drop(columns=existing_cols_to_drop)

# GENDER를 숫자로 변환 (M=1, F=0)
if 'GENDER' in df_features.columns:
    df_features['GENDER'] = df_features['GENDER'].map({'M': 1, 'F': 0})

# 타겟 및 민감 변수 정의
target_col = 'hospital_mortality'
sensitive_cols = ['ETHNICITY', 'INSURANCE']

# 모델 입력 변수 리스트 (타겟, 민감 변수, 식별자 제외)
model_input_features = [c for c in df_features.columns if c not in [target_col, 'SUBJECT_ID'] + sensitive_cols]
print(f"✅ 최종 모델 입력 변수 개수: {len(model_input_features)}")
print(f"   변수 목록 (67개 예상): {model_input_features}")

if len(model_input_features) != 67:
    print(f"   ⚠️ 경고: 변수 개수가 67개가 아닙니다! ({len(model_input_features)}개)")

# 결측치 처리 (Mean Imputation)
print("-> 결측치 처리 (Mean Imputation)...")
missing_threshold = 0.3
cols_to_drop = []
force_include_patterns = ['lactate', 'glucose']

for col in model_input_features:
    if any(pattern in col.lower() for pattern in force_include_patterns):
        continue
    if df_features[col].isnull().sum() / len(df_features) > missing_threshold:
        cols_to_drop.append(col)

if cols_to_drop:
    print(f"   제거될 변수 (Missing > {missing_threshold*100}%): {cols_to_drop}")
    df_features.drop(columns=cols_to_drop, inplace=True)
    # 제거 후 변수 목록 갱신
    model_input_features = [c for c in df_features.columns if c not in [target_col] + sensitive_cols]

# 평균값 보간
for col in model_input_features:
    if df_features[col].isnull().sum() > 0:
        mean_val = df_features[col].mean()
        df_features[col].fillna(mean_val, inplace=True)

# 데이터 분할 및 저장
# 데이터 저장 (전체 데이터)
save_dir = DATA_PATH
save_dir.mkdir(exist_ok=True, parents=True)

# 저장 (민감 변수 및 타겟 제외한 X)
# SUBJECT_ID는 각 파일에 모두 포함
X_all = df_features.drop(columns=sensitive_cols + [target_col]) # SUBJECT_ID 포함됨
y_all = df_features[[target_col, 'SUBJECT_ID']]
A_all = df_features[sensitive_cols + ['SUBJECT_ID']]

X_all.to_csv(save_dir / 'X_all.csv', index=False)
y_all.to_csv(save_dir / 'Y_all.csv', index=False)
A_all.to_csv(save_dir / 'A_all.csv', index=False)

print(f"   ✅ Saved all data to {save_dir}")

print("\n✨ 모든 데이터 전처리 완료!")
