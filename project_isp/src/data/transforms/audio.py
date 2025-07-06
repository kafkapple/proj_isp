class AudioTransform:
    def __init__(self, config):
        self.config = config
        self.transforms = []
        
        if config.augmentation.enabled:
            if config.augmentation.transforms.noise.enabled:
                self.transforms.append(
                    AddNoise(noise_level=config.augmentation.transforms.noise.noise_level)
                )
            
            if config.augmentation.transforms.volume.enabled:
                self.transforms.append(
                    RandomVolume(
                        min_gain=config.augmentation.transforms.volume.min_gain,
                        max_gain=config.augmentation.transforms.volume.max_gain
                    )
                )
            
            if config.augmentation.transforms.pitch_shift.enabled:
                self.transforms.append(
                    PitchShift(steps=config.augmentation.transforms.pitch_shift.steps)
                )
            
            if config.augmentation.transforms.time_stretch.enabled:
                self.transforms.append(
                    TimeStretch(rates=config.augmentation.transforms.time_stretch.rates)
                )
            
            if config.augmentation.transforms.spec_augment.enabled:
                self.transforms.append(
                    SpecAugment(
                        freq_mask_param=config.augmentation.transforms.spec_augment.freq_mask_param,
                        time_mask_param=config.augmentation.transforms.spec_augment.time_mask_param
                    )
                ) 