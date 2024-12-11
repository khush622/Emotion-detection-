import cv2
from sklearn.preprocessing import MinMaxScaler
import PIL
from PIL import Image
import re
%matplotlib inline
import os
import numpy as np


def fourier_deionising(path):
  # Load the image
  image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

  # Apply 2D FFT
  f_transform = np.fft.fft2(image)
  f_shift = np.fft.fftshift(f_transform)

  # Visualize the magnitude spectrum (optional)
  magnitude_spectrum = np.log(np.abs(f_shift) + 1)


  # Define a filter in the frequency domain to remove noise
  rows, cols = image.shape
  crow, ccol = rows // 3, cols // 2
  radius =15  # Adjust this parameter based on the level of noise to be removed
  mask = np.zeros((rows, cols), np.uint8)
  mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

  # Apply the filter in the frequency domain
  f_shift_filtered = f_shift * mask

  # Inverse FFT
  f_ishift = np.fft.ifftshift(f_shift_filtered)
  image_denoised = np.fft.ifft2(f_ishift)
  image_denoised = np.abs(image_denoised)
  return image_denoised


X_train=[]
X_test=[]
y_train=[]
y_test=[]
for dirname, _, filenames in os.walk('/kaggle/input/fer2013/train'):
        label_match = re.search(r'/([^/]+)$', dirname)
        label = label_match.group(1)
        for filename in filenames:
            path=(os.path.join(dirname, filename))
            
#             img = Image.open(path)

            X_train.append(fourier_deionising(path))
            y_train.append(label)
            
            
for dirname, _, filenames in os.walk('/kaggle/input/fer2013/test'):
        label_match = re.search(r'/([^/]+)$', dirname)
        label = label_match.group(1)
        for filename in filenames:
            path=(os.path.join(dirname, filename))
            X_test.append(fourier_deionising(path))
            y_test.append(label)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
y_train = pd.DataFrame(y_train, columns=['label'])
y_test  = pd.DataFrame(y_test, columns=['label'])


X_train_reshaped  = X_train.reshape((len(X_train), -1))
X_test_reshaped   = X_test.reshape((len(X_test), -1))
min_max_scaler    = MinMaxScaler()
X_train_scaled    = min_max_scaler.fit_transform(X_train_reshaped)
X_test_scaled     = min_max_scaler.transform(X_test_reshaped)
X_train_scaled    = X_train_scaled.reshape((len(X_train), 48, 48, 1))
X_test_scaled     = X_test_scaled.reshape((len(X_test), 48, 48, 1))
