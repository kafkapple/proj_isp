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

        csv_path = download_isear_dataset(cfg)
        
        print("Loading data...")
        dataset_cfg = cfg.data.datasets[cfg.data.name]
        
        df = pd.read_csv(
            csv_path,
            sep=dataset_cfg.separator,
            on_bad_lines='skip',
            encoding='utf-8',
            engine='python',
            quoting=3,
            dtype=str
        )
        
        required_columns = dataset_cfg.required_columns
        if not all(col in df.columns for col in required_columns):
            #  Column names contain the separator
            df.columns = [col.split(dataset_cfg.separator)[0] for col in df.columns]
        
        # Select only the required columns
        self.df = df[required_columns]
        
        # Normalize emotion labels
        self.df['EMOT'] = self.df['EMOT'].str.lower()
        
        # Drop missing values and duplicates
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        
        # Print initial class distribution
        self._print_class_distribution("Initial class distribution:", self.df)
        
        # Stratified sampling only when n_samples is not -1
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
        """Print class distribution"""
        dist = df[self.cfg.data.column_mapping.emotion].value_counts()
        total = len(df)
        
        print(f"\n{title}")
        print("-" * 40)
        for emotion, count in dist.items():
            percentage = (count / total) * 100
            print(f"{emotion:10}: {count:5} ({percentage:5.1f}%)")
    
    def _stratified_sample(self, df: pd.DataFrame, n_samples: int, stratify_col: str):
        """Perform stratified sampling"""
        if n_samples >= len(df):
            return df
            
        return df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(n_samples * len(x) / len(df))),
                random_state=42
            )
        ).reset_index(drop=True)
        
    def split_data(self):
        """Perform stratified train/val split"""
        print("\nSplitting data...")
        
        # train/val split (stratified)
        train_df, val_df = train_test_split(
            self.df, 
            test_size=self.cfg.data.val_size,  # validation size directly used
            stratify=self.df[self.cfg.data.column_mapping.emotion],
            random_state=self.cfg.general.random_state
        )
        
        # Print class distribution after splitting
        print("\nClass distributions after splitting:")
        self._print_class_distribution("Train set:", train_df)
        self._print_class_distribution("Validation set:", val_df)
        
        return train_df, val_df 