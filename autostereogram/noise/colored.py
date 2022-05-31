import numpy.fft as fft
import numpy as np


def create_pink_noise(patch_size):
    white_noise = np.random.uniform(0, 1, (patch_size, patch_size))
    fourier_transformed = fft.fft2(white_noise)
    f_y = fft.fftfreq(fourier_transformed.shape[0],d=1/72)
    f_x = fft.fftfreq(fourier_transformed.shape[1], d=1/72)
    pink_transformed = np.copy(fourier_transformed)
    for y in range(fourier_transformed.shape[0]):
        for x in range(fourier_transformed.shape[1]):
            if f_x[x] == 0 and f_y[y] == 0:
                continue
            # elif f_x[x] == 0:
            #     pink_transformed[y, x] = pink_transformed[y, x] / (f_y[y] ** 2)
            # elif f_y[y] == 0:
            #     pink_transformed[y, x] = pink_transformed[y, x] / (f_x[x] ** 2)
            else:
                pink_transformed[y, x] = pink_transformed[y, x] / np.sqrt(f_y[y] ** 2 + f_x[x] ** 2)
    pink_noise = fft.ifft2(pink_transformed)

    # Normalize the noise patch to 0-255
    pink_noise = pink_noise - pink_noise.min()
    pink_noise = (pink_noise / pink_noise.max()) * 255

    return pink_noise.astype(int)
