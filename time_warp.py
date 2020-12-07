import numpy as np
from tensorflow_addons.image import sparse_image_warp

def time_warp(mel_spectrogram,W=30):
    
    """
    Given a Mel-scale spectrogram with shape v x tau, time
    warping is done as follows:
    
    1. Pick a column on the time axis at random from column number W to column
       number tau - W, where W is the time warp parameter that is chosen
       beforehand. In the original paper, W ranges from 0 to 80. Let the chosen
       column number be alpha. This means that the Mel-spectrogram is now
       partitioned into two windows defined on the time axis as:
           
       1st window:
           column 0 --> column alpha
           
       2nd window:
           column alpha + 1 --> column tau
    
    2. The 1st window is to be warped to the left or right by a distance w that
       is sampled from a discrete uniform distribution with a support of [-W,W].
       The 2nd window is warped in the opposite direction by the same amount
       as the 1st window.
    3. To perform this warping, control points are placed on each pixel in the
       alpha column. The final locations of these pixels after warping are also
       specified. Finally, the Tensorflow Addons function
       tensorflow_addons.image.sparse_image_warp() is then used to perform time
       warping.
    
    The original SpecAugment paper can be found here:
        
    https://arxiv.org/abs/1904.08779
        
    """
    
    # numpy arrays are mutable, so need to make copy to avoid modifying
    # original array
    
    mel_spec = mel_spectrogram.copy()
    
    # width and height of Mel-scale spectrogram. Height is equivalent to number
    # of Mels and width is equivalent to number of time steps.
    # v and tau are used to match the notation used in the original paper
    
    v, tau = mel_spec.shape
    
    # initialize random number generator
    
    rng = np.random.default_rng()
    
    # alpha is the column to be warped
    
    alpha = rng.integers(W,tau - W,
                         endpoint = True)

    # each row of this array will contain the locations of the points to be
    # warped. More precisely, it contains the coordinates of the alpha column,
    # such that each pixel in the alpha column has its own coordinate

    src_ctrl_pt_loc = np.zeros((v,2),dtype=np.float32)
    
    # each row of the alpha column
    
    src_ctrl_pt_loc[:,0] = np.arange(v)
    
    # the alpha column
    
    src_ctrl_pt_loc[:,1] = alpha

    # add a batch dimension for the sparse_image_warp function to be used later

    src_ctrl_pt_loc = np.expand_dims(src_ctrl_pt_loc, 0)
    
    # sample w from a discrete uniform distribution
    
    w = rng.integers(-W,W,
                     endpoint = True)

    # each row of this array will contain the locations of the destinations that
    # the points in src_ctrl_pt_loc will be warped to. These coordinates
    # correspond to the target column

    dst_ctrl_pt_loc = np.zeros((v,2),dtype=np.float32)
    
    # each row of the target column
    
    dst_ctrl_pt_loc[:,0] = np.arange(v)
    
    # the target column
    
    dst_ctrl_pt_loc[:,1] = alpha + w

    # add a batch dimension for the sparse_image_warp function to be used later

    dst_ctrl_pt_loc = np.expand_dims(dst_ctrl_pt_loc, 0)

    # add batch and channel dimensions for the sparse_image_warp function to be
    # used later

    mel_spec = np.reshape(mel_spec,(1,*mel_spec.shape,1))
    
    """    
    the sparse_image_warp function returns a tuple, where the first element
    is the warped Mel-scale spectrogram and the second element is the dense
    flow field that is produced by interpolation. Only the warped Mel-scale
    spectrogram is of interest. Also, because the function returns a
    tensorflow tensor, it is converted to a numpy array using the .numpy()
    method.
    
    the num_boundary_points argument is set to be greater than 0 such that
    frequency content is not lost while warping by constraining the warping
    transformation.
    """
    
    warped_spec = sparse_image_warp(
                        image = mel_spec,
                        source_control_point_locations = src_ctrl_pt_loc,
                        dest_control_point_locations = dst_ctrl_pt_loc,
                        interpolation_order = 2,
                        regularization_weight = 0.0,
                        num_boundary_points = 1)[0].numpy()
    
    # remove redundant dimensions
    
    warped_spec = np.squeeze(warped_spec)
    
    return warped_spec,alpha,w

if __name__ == '__main__':
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    
    x,sr = librosa.load(path = 'test.wav',
                        sr = None,
                        mono = True)

    mel_spec = librosa.feature.melspectrogram(y = x,
                                              sr = sr,
                                              n_fft = 2048,
                                              hop_length = 32,
                                              power = 2.0,
                                              n_mels = 256)
    
    warped_spec,alpha,w = time_warp(mel_spec,W=30)
   
    # show original spectrogram and warped spectrogram
    
    # original spectrogram
        
    plt.subplot(2,1,1)
    
    # convert to dB
    
    mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
    
    # plot vertical line showing the column that is warped
    
    mel_spec[:,alpha] = 0
    
    # display the original spectrogram
    
    librosa.display.specshow(mel_spec,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    
    # warped spectrogram
    
    plt.subplot(2,1,2)
    
    # convert to dB
    
    warped_spec = librosa.power_to_db(warped_spec,ref=np.max)
    
    # plot vertical line to show warping
    
    warped_spec[:,alpha + w] = 0
    
    # display the warped spectrogram
    
    librosa.display.specshow(warped_spec,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    
    plt.tight_layout()