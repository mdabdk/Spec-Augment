import numpy as np

def time_mask(mel_spectrogram, T = 30, num_masks = 1):
    
    # numpy arrays are mutable, so need to make copy to avoid modifying
    # original array
    
    mel_spec = mel_spectrogram.copy()
    
    # number of time steps in spectrogram
    
    tau = mel_spec.shape[1]
    
    # initialize random number generator
        
    rng = np.random.default_rng()
    
    # generate num_masks time masks
    
    for _ in range(0,num_masks):
        
        t = rng.integers(0,T,
                         endpoint = True)
        
        t_0 = rng.integers(0,tau - t,
                           endpoint = True)
        
        mel_spec[:, t_0:t_0 + t] = np.mean(mel_spec)
    
    return mel_spec

if __name__ == '__main__':
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    
    # load test audio file
    
    x,sr = librosa.load(path = 'test.wav',
                        sr = None,
                        mono = True)
    
    mel_spec = librosa.feature.melspectrogram(y = x,
                                              sr = sr,
                                              n_fft = 2048,
                                              hop_length = 32,
                                              power = 2.0,
                                              n_mels = 256)
    
    mel_spec2 = time_mask(mel_spec,
                          T = 20,
                          num_masks = 2)
    
    # original spectrogram
    
    plt.subplot(2,1,1)
    
    # convert to dB
    
    mel_spec = librosa.power_to_db(mel_spec,
                                   ref = np.max)
    
    # display the original spectrogram
    
    librosa.display.specshow(mel_spec,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    
    # masked spectrogram
    
    plt.subplot(2,1,2)
    
    # convert to dB
    
    mel_spec2 = librosa.power_to_db(mel_spec2,
                                    ref = np.max)
    
    # display the masked spectrogram
    
    librosa.display.specshow(mel_spec2,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    
    plt.tight_layout()