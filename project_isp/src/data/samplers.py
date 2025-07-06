class BalancedBatchSampler(Sampler):
    """각 배치에 모든 클래스가 균등하게 포함되도록 함"""
    def __init__(self, dataset, batch_size):
        self.labels = dataset.labels
        self.num_classes = len(np.unique(self.labels))
        self.batch_size = batch_size
        self.samples_per_class = batch_size // self.num_classes
        
        # 클래스별 인덱스
        self.class_indices = [np.where(self.labels == i)[0] for i in range(self.num_classes)] 