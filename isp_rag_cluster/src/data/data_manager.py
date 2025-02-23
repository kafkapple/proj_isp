import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from .download_isear import download_isear_dataset

class DataManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print("\nEmotion Classes:")
        self.emotion_map = {str(i+1): emotion for i, emotion in enumerate(cfg.emotions.classes)}
        for id, emotion in self.emotion_map.items():
            print(f"ID {id}: {emotion}")
        print("-" * 80)
        
        # 데이터 다운로드 (필요한 경우)
        csv_path = download_isear_dataset(cfg)
        
        print("Loading data...")
        # 데이터셋 설정 가져오기
        dataset_cfg = cfg.data.datasets[cfg.data.name]
        
        # 원본 데이터 로드
        df = pd.read_csv(
            csv_path,
            sep=dataset_cfg.separator,
            on_bad_lines='skip',
            encoding='utf-8',
            engine='python',
            quoting=3,
            dtype=str
        )
        
        # 필요한 컬럼 추출 및 전처리
        required_columns = dataset_cfg.required_columns
        if not all(col in df.columns for col in required_columns):
            # 컬럼명에 구분자가 포함된 경우 처리
            df.columns = [col.split(dataset_cfg.separator)[0] for col in df.columns]
        
        # 필요한 컬럼만 선택
        self.df = df[required_columns]
        
        # 감정 레이블 정규화
        self.df['EMOT'] = self.df['EMOT'].str.lower()
        
        # 결측치 및 중복 제거
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        
        # 초기 클래스 분포 출력
        self._print_class_distribution("Initial class distribution:", self.df)
        
        # n_samples가 -1이 아닐 때만 stratified 샘플링
        if cfg.data.n_samples > 0:
            self.df = self._stratified_sample(
                self.df, 
                n_samples=cfg.data.n_samples, 
                stratify_col=cfg.data.column_mapping.emotion
            )
            print("\nAfter sampling:")
            self._print_class_distribution("Sampled class distribution:", self.df)
            
        print(f"\nTotal samples: {len(self.df)}")
        
    def _print_class_distribution(self, title: str, df: pd.DataFrame):
        """클래스 분포 출력"""
        dist = df[self.cfg.data.column_mapping.emotion].value_counts()
        total = len(df)
        
        print(f"\n{title}")
        print("-" * 40)
        for emotion, count in dist.items():
            percentage = (count / total) * 100
            print(f"{emotion:10}: {count:5} ({percentage:5.1f}%)")
    
    def _stratified_sample(self, df: pd.DataFrame, n_samples: int, stratify_col: str):
        """층화 샘플링 수행"""
        if n_samples >= len(df):
            return df
            
        return df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(n_samples * len(x) / len(df))),
                random_state=42
            )
        ).reset_index(drop=True)
        
    def split_data(self):
        """stratified train/val split 수행"""
        print("\nSplitting data...")
        
        # train/val 분할 (stratified)
        train_df, val_df = train_test_split(
            self.df, 
            test_size=self.cfg.data.val_size,  # validation size 직접 사용
            stratify=self.df[self.cfg.data.column_mapping.emotion],
            random_state=42
        )
        
        # 분할 결과의 클래스 분포 출력
        print("\nClass distributions after splitting:")
        self._print_class_distribution("Train set:", train_df)
        self._print_class_distribution("Validation set:", val_df)
        
        return train_df, val_df 