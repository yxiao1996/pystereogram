import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.draw import disk
import time

def draw_circle(image, y, x, r=5):
    [rr, cc] = disk((y, x), r)
    image[rr, cc] = 0

if __name__ == "__main__":
    rand_size = 64
    cube_image = mpimg.imread('./example/inputs/cat.png')
    image_data = np.array(color.rgb2gray(cube_image))

    # Scale the pixel values and reduce the depth
    image_data = image_data.max() - image_data
    image_data = image_data - image_data.min()
    image_data = np.floor(image_data / image_data.max() * rand_size / 3)
    image_data = image_data.astype(int)
    # Create a random seed to build the image from
    rand_data = np.floor(np.random.uniform(0, 255, (rand_size, rand_size))).astype(int)

    [image_height, image_width] = image_data.shape
    [rand_height, rand_width] = rand_data.shape
    start_time = time.time()
    for y in range(image_height):
        y_index = np.mod(y, rand_height - 1)
        # Start from the right most copt of the random seed
        for x in range(image_width - 1, image_width - rand_width - 1, -1):
            x_index = np.mod(x - image_data[y, x], rand_width - 1)
            image_data[y, x] = rand_data[y_index, x_index]

        # Recalculate all the remaining pixels on the line
        for x in range(image_width - rand_width - 1, 0, -1):
            index = x + rand_width - image_data[y, x]
            image_data[y, x] = image_data[y, index]
    draw_circle(image_data, image_height * 19 / 20, image_width / 2 - rand_size / 2)
    draw_circle(image_data, image_height * 19 / 20, image_width / 2 + rand_size / 2)
    print(time.time() - start_time)
    plt.imsave('./example/outputs/cat.jpg', color.gray2rgb(image_data / 255))