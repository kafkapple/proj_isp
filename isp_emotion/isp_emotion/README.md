# isp_emotion

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `config.yaml`: Main configuration
  - Logging settings
  - Experiment tracking (Weights & Biases)
  - Debug options
- `dataset/ravdess.yaml`: Dataset-specific settings
  - Audio preprocessing parameters
  - Class weights
  - Data augmentation settings
  - Split ratios (70/15/15)
- `train/train.yaml`: Training parameters
  - Optimizer settings (AdamW)
  - Learning rate scheduling
  - Loss function configuration
  - Mixup augmentation
- `model/wav2vec.yaml`: Model architecture settings
  - Wav2Vec 2.0 configuration
  - Feature extractor architecture
  - Layer freezing options

## Model Architecture

1. **Base Model**: Wav2Vec 2.0 (facebook/wav2vec2-base)
   - Pretrained on 960 hours of LibriSpeech
   - Gradient checkpointing enabled for memory efficiency
   - Configurable layer freezing for transfer learning

2. **Feature Extractor**:
   - Linear(768 → 512) → BatchNorm → ReLU → Dropout
   - Linear(512 → 256) → BatchNorm → ReLU → Dropout
   - Adaptive feature extraction for emotion recognition

3. **Classifier Head**: 
   - Linear(256 → num_classes)
   - Optimized for emotion classification

## Training Strategy

- **Three-tier Learning Rate**:
  - Frozen Wav2Vec layers: lr * 0.01
  - Unfrozen Wav2Vec layers: lr * 0.1
  - Classifier layers: lr (base learning rate)

- **Optimization**:
  - AdamW optimizer with weight decay
  - Cosine annealing with warm restarts
  - Gradient clipping for stability

- **Regularization**:
  - Mixup augmentation with clean metrics calculation
  - Dropout in feature extractor
  - Label smoothing
  - Class weights for imbalanced data

- **Loss Function**:
  - Focal loss with gamma=1.0
  - Class weights support
  - Label smoothing (0.05)

## Results

Performance metrics on RAVDESS dataset:
- Accuracy: X%
- Macro F1: X%
- Weighted F1: X%

(Note: Replace X with actual metrics from your best model)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook AI Research for Wav2Vec 2.0
- RAVDESS dataset creators
- PyTorch Lightning team