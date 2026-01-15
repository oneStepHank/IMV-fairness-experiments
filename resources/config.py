from pathlib import Path

# --- 데이터 경로 설정 ---
# MIMIC-III 원본 데이터가 있는 디렉토리 (CSV 파일들)
MIMIC_DIR = Path('../../MIMIC-III') 

# 전처리된 데이터가 저장될 디렉토리
DATA_PATH = Path('./data/processed_data')

# --- 기타 설정 ---
CHUNK_SIZE = 1_000_000
