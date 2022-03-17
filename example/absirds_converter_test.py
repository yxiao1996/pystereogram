import numpy as np
from autostereogram.sirds_converter import SirdsConverter
from skimage import color
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    source_image = mpimg.imread('./inputs/cube.jpg')
    if len(source_image.shape) == 3:
        source_image = color.rgb2gray(source_image)
    image_data = np.array(source_image * 255, dtype=int)

    converter = SirdsConverter()
    start_time = time.time()
    image_data = image_data.max() - image_data
    image_data = converter.convert_depth_to_stereogram_with_sird(image_data, True, 0.5).astype(np.uint8)
    print(time.time() - start_time)
    plt.imsave('./outputs/cube6.jpg', color.gray2rgb(image_data))