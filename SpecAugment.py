import numpy as np
from tensorflow_addons.image import sparse_image_warp
import librosa.display
import matplotlib.pyplot as plt
import random

class SpecAugment:
    
    def __init__(self,
                 num_aug,
                 p,
                 num_time_masks,
                 num_frequency_masks,
                 time_warp_param = 30,
                 time_mask_param = 30,
                 frequency_mask_param = 30,
                 num_masks_random = False):
        
        # number of Mel-scale spectrogram augmentations to perform
        
        self.num_aug = num_aug
        
        """
        The following list contains the proportions of the total number of augmentations
        that are time warped (TW), time masked (TM), and frequency masked (FM). These
        transformations are considered as random variables with associated probability
        distributions P(TW), P(TM), and P(FM) respectively. These are Bernoulli random
        variables, where each transformation can either be applied (1) or not applied
        (0).
        
        The first element of the list corresponds to the proportion of the total number
        of augmentations that are time warped, or P(TW = 1). The second element of the
        list corresponds to the proportion of the total number of augmentations that are
        time masked, or P(TM = 1). The third element of the list corresponds to the
        proportion of the total number of augmentations that are frequency masked, or
        P(FM = 1). For example, if:
            
        p = [0.4,0.6,0.7]
        
        Then this means that of the total number of num_aug augmentations,
        40% will be time warped, 60% will be time masked, and 70% will be
        frequency masked. Note that:
        
        P(TW = 0) = 1 - P(TW = 1)
        P(TM = 0) = 1 - P(TM = 1)
        P(FM = 0) = 1 - P(FM = 1)
        
        IMPORTANT NOTE: Because the probability of not time warping, not time
        masking, and not frequency masking a spectrogram is set to 0, then the
        sum of these probabilities MUST be at least 1.5. In other words, 
        sum(p) > 1.5.
        """
        
        self.p = p
        
        """
        The possible transformations that can be applied are shown in the table below.
        TW stands for time warping, TM stands for time masking, and FM stands for
        frequency masking. A 1 indicates that the corresponding transformation is
        applied, while a 0 indicates that the corresponding transformation is not
        applied.
        
        TW  TM  FM
        -----------
        0   0   0
        0   0   1
        0   1   0
        0   1   1
        1   0   0
        1   0   1
        1   1   0
        1   1   1
        
        Since these transformations can be applied in combinations, then a joint
        distribution for these transformations P(TW,TM,FM) also exists. However, these
        transformations are assumed to be independent from each other, such that the
        occurence of one transformation does not imply anything about the occurence of
        another transformation. This means that:
        
        P(TW,TM,FM) = P(TW) x P(TM) x P(FM)
        
        To avoid no augmentations occuring at all, which is when TW = 0 and TM = 0 and
        FM = 0, then the joint probability of these events should be set to 0:
            
        P(TW = 0,TM = 0,FM = 0) = 0
        
        However, the probability mass associated with P(TW = 0,TM = 0,FM = 0) must now
        be re-distributed to the rest of the probability distribution. Since there
        are 7 outcomes in total, then the following probability mass is re-distributed:
            
        P(TW = 0,TM = 0,FM = 0) / 7 = P(TW = 0) x P(TM = 0) x P(FM = 0) / 7
        
        This is equivalent to re-distributing:
            
        (1 - P(TW = 1)) x (1 - P(TM = 1)) x (1 - P(FM = 1)) / 7
        
        Re-distribution is done by adding this probability mass to the probability mass
        of each of the other outcomes.
        """

        self.redistribution = (1-p[0])*(1-p[1])*(1-p[2])/7
        
        # the joint distribution P(TW,TM,FM) = P(TW) x P(TM) x P(FM) is defined here.
        # Note that the sum of all probability masses must be equal to 1

        self.joint = [0, # P(TW = 0,TM = 0,FM = 0)
                      (1-p[0])*(1-p[1])*( p[2] ) + self.redistribution, # P(TW = 0,TM = 0,FM = 1)
                      (1-p[0])*( p[1] )*(1-p[2]) + self.redistribution, # P(TW = 0,TM = 1,FM = 0)
                      (1-p[0])*( p[1] )*( p[2] ) + self.redistribution, # P(TW = 0,TM = 1,FM = 1)
                      ( p[0] )*(1-p[1])*(1-p[2]) + self.redistribution, # P(TW = 1,TM = 0,FM = 0)
                      ( p[0] )*(1-p[1])*( p[2] ) + self.redistribution, # P(TW = 1,TM = 0,FM = 1)
                      ( p[0] )*( p[1] )*(1-p[2]) + self.redistribution, # P(TW = 1,TM = 1,FM = 0)
                      ( p[0] )*( p[1] )*( p[2] ) + self.redistribution] # P(TW = 1,TM = 1,FM = 1)
        
        # initialize random number generator
        
        self.rng = np.random.default_rng()
        
        # time warping parameter W. See the time_warp() method for details
        
        self.W = time_warp_param
        
        # time masking parameter T. See the time_mask() method for details
        
        self.T = time_mask_param
        
        # frequency masking parameter F. See the frequency_mask() method for
        # details
        
        self.F = frequency_mask_param
        
        # the number of masks to be applied to the time axis of the Mel-scale
        # spectrogram
        
        self.num_time_masks = num_time_masks
        
        # the number of masks to be applied to the frequency axis of the
        # Mel-scale spectrogram
        
        self.num_frequency_masks = num_frequency_masks
        
        """
        if this is set to true, then the number of time masks that are applied
        will be sampled from a discrete uniform distribution with a support
        of (1,T), while the number of frequency masks that are applied will
        be sampled from a discrete uniform distribution with a support of
        (1,F). The number of time masks and frequency masks that are applied
        will be random for each augmentation.
        
        Otherwise, the number of time masks that are applied will be fixed
        and equal to num_time_masks and the number of frequency masks will be
        fixed and equal to num_frequency_masks.
        """
        
        self.num_masks_random = num_masks_random
        
    def time_warp(self,log_mel_spec):
    
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
            
        y = log_mel_spec.copy()
        
        # width and height of Mel-scale spectrogram. Height is equivalent to number
        # of Mels and width is equivalent to number of time steps.
        # v and tau are used to match the notation used in the original paper
        
        v, tau = y.shape
        
        # alpha is the column to be warped
        
        alpha = self.rng.integers(self.W,tau - self.W,
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
        
        w = self.rng.integers(-self.W,self.W,
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
    
        y = np.reshape(y,(1,*y.shape,1))
        
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
        
        y = sparse_image_warp(image = y,
                              source_control_point_locations = src_ctrl_pt_loc,
                              dest_control_point_locations = dst_ctrl_pt_loc,
                              interpolation_order = 2,
                              regularization_weight = 0.0,
                              num_boundary_points = 1)[0].numpy()
        
        # remove redundant dimensions
        
        y = np.squeeze(y)
        
        return y
    
    def time_mask(self,log_mel_spec):
        
        # numpy arrays are mutable, so need to make copy to avoid modifying
        # original array
            
        y = log_mel_spec.copy()
        
        # if a random number of time masks are to be applied
        
        if self.num_masks_random:
            num_masks = self.rng.integers(1,self.num_time_masks,
                                          endpoint = True)
        else:
            num_masks = self.num_time_masks
        
        # number of time steps in spectrogram
        
        tau = y.shape[1]
        
        # generate num_masks time masks
        
        for _ in range(0,num_masks):
            
            t = self.rng.integers(0,self.T,
                                  endpoint = True)
            
            t_0 = self.rng.integers(0,tau - t,
                                    endpoint = True)
            
            y[:, t_0:t_0 + t] = np.mean(y)
        
        return y
        
    def frequency_mask(self,log_mel_spec):
        
        # numpy arrays are mutable, so need to make copy to avoid modifying
        # original array
            
        y = log_mel_spec.copy()
        
        # if a random number of frequency masks are to be applied
                
        if self.num_masks_random:
            num_masks = self.rng.integers(1,self.num_frequency_masks,
                                            endpoint = True)
        else:
            num_masks = self.num_frequency_masks
        
        # number of Mels in spectrogram
        
        v = y.shape[0]
        
        # generate num_masks frequency masks
        
        for _ in range(0,num_masks):
            
            f = self.rng.integers(0,self.F,
                             endpoint = True)
            
            f_0 = self.rng.integers(0,v - f,
                               endpoint = True)
            
            y[f_0:f_0 + f,:] = np.mean(y)
        
        return y
        
    def __call__(self,log_mel_spec):
        
        # lists to store augmentations
        
        augmentations = []
        
        # track number of each augmentation
            
        time_warped = 0
        time_masked = 0
        frequency_masked = 0
        
        # apply transformations randomly
        
        for _ in range(self.num_aug):
            
            # sample categorical random variable from 0 - 7
            
            x = self.rng.multinomial(n = 1,
                                     pvals = self.joint,
                                     size = 1)
            
            x = np.nonzero(x)[1]
            
            if x == 1:
                
                y = self.frequency_mask(log_mel_spec)
                frequency_masked += 1
            
            if x == 2:
                
                y = self.time_mask(log_mel_spec)
                time_masked += 1
            
            if x == 3:
                
                y = self.time_mask(log_mel_spec)
                y = self.frequency_mask(y)
                time_masked += 1
                frequency_masked += 1
            
            if x == 4:
                
                y = self.time_warp(log_mel_spec)
                time_warped += 1
                
            if x == 5:
                
                y = self.time_warp(log_mel_spec)
                y = self.frequency_mask(y)
                time_warped += 1
                frequency_masked += 1
            
            if x == 6:
                
                y = self.time_warp(log_mel_spec)
                y = self.time_mask(y)
                time_warped += 1
                time_masked += 1
            
            if x == 7:
                
                y = self.time_warp(log_mel_spec)
                y = self.time_mask(y)
                y = self.frequency_mask(y)
                time_warped += 1
                time_masked += 1
                frequency_masked += 1
            
            # save the augmentation
            
            augmentations.append(y)
        
        proportions = [time_warped/self.num_aug,
                       time_masked/self.num_aug,
                       frequency_masked/self.num_aug]
        
        return augmentations,proportions
    
    def show_aug(self,log_mel_spec,augmentations):
        
        # pick a transformed log Mel spectrogram at random from the list of
        # augmentations
        
        aug = random.choice(augmentations)
        
        # display the original spectrogram
    
        plt.subplot(2,1,1)
        
        librosa.display.specshow(log_mel_spec,
                                 x_axis='time',
                                 y_axis='mel')
        
        plt.colorbar(format='%+2.0f dB')
        
        plt.title('Original Mel spectrogram')
        
        # display the transformed spectrogram
        
        plt.subplot(2,1,2)
        
        librosa.display.specshow(aug,
                                 x_axis='time',
                                 y_axis='mel')
        
        plt.colorbar(format='%+2.0f dB')
        
        plt.title('Transformed Mel spectrogram')
        
        plt.tight_layout()
        

if __name__ == '__main__':
    
    import librosa
    
    x,sr = librosa.load(path = 'test.wav',
                        sr = None,
                        mono = True)

    mel_spec = librosa.feature.melspectrogram(y = x,
                                              sr = sr,
                                              n_fft = 2048,
                                              hop_length = 32,
                                              power = 2.0,
                                              n_mels = 256)
    
    log_mel_spec = librosa.power_to_db(mel_spec,
                                       ref = np.max)
    
    spec_augment = SpecAugment(num_aug = 100,
                               p = [0.4,0.4,0.4],
                               time_warp_param = 30,
                               time_mask_param = 20,
                               frequency_mask_param = 30,
                               num_time_masks = 2,
                               num_frequency_masks = 3,
                               num_masks_random = True)
    
    augmentations,proportions = spec_augment(log_mel_spec)
    
    spec_augment.show_aug(log_mel_spec,augmentations)
            