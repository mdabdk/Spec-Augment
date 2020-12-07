import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow_addons.image import sparse_image_warp

x,sr = librosa.load(path = 'test.wav',
                    sr = None,
                    mono = True)

mel_spec = librosa.feature.melspectrogram(y = x,
                                          sr = sr,
                                          n_fft = 2048,
                                          hop_length = 32,
                                          power = 2.0,
                                          n_mels = 256)

# mel_spec = np.arange(10000).reshape((100,100))

# height is length of frequency axis, and width is length of time axis.
# tau is the same notation used in the original paper.

mel_spec_height, tau = mel_spec.shape

"""
Given a Mel-scale spectrogram with shape mel_spec_height x tau, time warping is
done as follows:

1. Pick a column on the time axis at random from column number W to column
   number tau - W, where W is the time warp parameter that is chosen beforehand.
   In the original paper, W ranges from 0 to 80. Let the chosen column number
   be alpha. This means that the Mel-spectrogram is now partitioned into two
   windows defined by the coordinates:
       
   1st window:
       row 0, column 0
       row mel_spec_height, column 0
       row 0, column alpha
       row mel_spec_height, column alpha
       
   2nd window:
       row 0, column alpha+1
       row mel_spec_height, column alpha+1
       row 0, column tau
       row mel_spec_height, column tau

2. The 1st window is to be warped to the left or right by a distance w that is
   sampled from a discrete uniform distribution with a support of [-W,W].
3. According to the original paper, to perform this warping, 6 control points
   are placed on the edges of the window. 4 of these control points will be
   placed on the corners of the window, and the other 2 will be placed at the
   mid-points of the vertical edges of the window. However, the implementation
   used here uses a different set of control points. This does not affect the
   outcome.
5. The Tensorflow Addons function tensorflow_addons.image.sparse_image_warp()
   is then used to perform time warping.
"""

# define the time warp parameter W

W = 20

# sample alpha. Note that this does not need to be a uniform distribution

alpha = np.random.randint(W,tau - W)

# the control points only need to be defined along the alpha column that
# separates the two windows

# defining the vertical coordinates for the control points along the alpha
# column

src_ctr_pt_freq = np.arange(mel_spec_height)

# defining the horizontrol coordinates for the control points along the alpha
# column

src_ctr_pt_time = np.ones_like(src_ctr_pt_freq) * alpha

# form the control points

src_ctr_pts = np.stack((src_ctr_pt_freq,src_ctr_pt_time),
                       axis = -1).astype(np.float32)

# src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

# sample w

w = 20#np.random.randint(-W,W)
dest_ctr_pt_freq = src_ctr_pt_freq
dest_ctr_pt_time = src_ctr_pt_time + w
dest_ctr_pts = np.stack((dest_ctr_pt_freq, dest_ctr_pt_time),
                        axis=-1).astype(np.float32)
# dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

# warp
source_control_point_locations = np.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
dest_control_point_locations = np.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)
mel_spec = np.reshape(mel_spec,(1,*mel_spec.shape,1))

warped_spec, _ = sparse_image_warp(mel_spec,
                                    source_control_point_locations,
                                    dest_control_point_locations)

# def show_spec(mel_spec):
#     spec = librosa.power_to_db(mel_spec[0,:,:,0],ref=np.max)
#     spec = ((spec - spec.min()) * (1/(spec.max() - spec.min()) * 255)).astype('uint8')
#     spec = Image.fromarray(spec)
#     draw = ImageDraw.Draw(spec) 
#     draw.line((0,alpha,mel_spec_height,alpha), fill=0)
#     spec.show()

# show_spec(warped_image)

# plt.figure(figsize=(10, 4))
ax1 = plt.subplot(2,1,1)

# spec[:,alpha] = 0
# librosa.display.specshow(spec,y_axis='mel',x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.vlines(x=alpha,ymin=0,ymax=mel_spec.shape[1])
# ax2 = plt.subplot(2,1,2)
# spec = librosa.power_to_db(warped_image[0,:,:,0],ref=np.max)
# spec[:,alpha+w] = 0
# librosa.display.specshow(spec,y_axis='mel',x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()

plt.subplot(2,1,1)
mel_spec = librosa.power_to_db(mel_spec[0,:,:,0],ref=np.max)
mel_spec[:,alpha] = 0
plt.imshow(mel_spec)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.subplot(2,1,2)
warped_spec = warped_spec.numpy()
warped_spec = librosa.power_to_db(warped_spec[0,:,:,0],ref=np.max)
warped_spec[:,alpha+w] = 0
plt.imshow(warped_spec)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()