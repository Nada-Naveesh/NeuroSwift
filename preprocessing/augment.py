import numpy as np

def sliding_window_augmentation(signal, window_size=500, stride=75, sampling_rate=160):
    window_samples = int(window_size * sampling_rate / 1000)
    stride_samples = int(stride * sampling_rate / 1000)
    
    windows = []
    for start in range(0, signal.shape[-1] - window_samples + 1, stride_samples):
        window = signal[:, start:start + window_samples]
        windows.append(window)
    
    return windows

def apply_window_augmentation(data, labels, window_size=500, stride=75, sampling_rate=160):
    augmented_data = []
    augmented_labels = []
    
    for idx, (signal, label) in enumerate(zip(data, labels)):
        windows = sliding_window_augmentation(signal, window_size, stride, sampling_rate=sampling_rate)
        augmented_data.extend(windows)
        augmented_labels.extend([label] * len(windows))
    
    return np.array(augmented_data), np.array(augmented_labels)
