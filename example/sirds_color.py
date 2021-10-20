import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.draw import disk
from skimage.transform import resize
import time

DPI = 72
mu = 1 / 3.0
E = np.round(2.5 * DPI)
def separation(z):
    return np.round((1 - mu * z) * E / (2 - mu * z))
far = separation(0)

def draw_circle(image, y, x, r=5):
    [rr, cc] = disk((y, x), r)
    image[rr, cc, :] = 0

def main():
    depth_image = mpimg.imread('inputs/cat-d.png')
    color_image = mpimg.imread('inputs/cat-c.jpeg')
    color_image = resize(color_image, (depth_image.shape[0], depth_image.shape[1]), anti_aliasing=True)
    image_data = np.array(color.rgb2gray(depth_image))

    # Scale the pixel values to between 0 and 1
    image_data = image_data.max() - image_data
    image_data = image_data - image_data.min()
    image_data = image_data / image_data.max()
    image_data = image_data.astype(float)
    [image_height, image_width] = image_data.shape
    z = image_data
    stereogram = np.zeros((image_height, image_width, 3))
    start_time = time.time()
    for y in range(image_height):
        pix = np.zeros((image_width, 3))
        same = np.zeros(image_width, dtype=int)

        for x in range(image_width):
            same[x] = x

        for x in range(image_width):
            s = separation(z[y, x])
            left = int(x - s / 2)
            right = int(left + s)
            if (0 <= left and right < image_width):
                t = 1
                # while True:
                #     zt = z[y, x] + 2 * (2 - mu * z[y, x]) * t / (mu * E)
                #     visible = (z[y, x-t] < zt and z[y, x+t] < zt)
                #     t = t + 1
                #     if not (visible and zt < 1):
                #         break
                visible = True
                if visible:
                    l = same[left]
                    while (l != left and l != right):
                        if (l < right):
                            left = l
                            l = same[left]
                        else:
                            same[left] = right
                            left = right
                            l = same[left]
                            right = l
                    same[left] = right
        for x in range(image_width - 1, -1, -1):
            if (same[x] == x):
                pix[x, :] = color_image[y, x, :]
            else:
                pix[x, :] = pix[int(same[x]), :]
        stereogram[y, :] = pix
    draw_circle(stereogram, image_height * 19 / 20, image_width / 2 - far / 2)
    draw_circle(stereogram, image_height * 19 / 20, image_width / 2 + far / 2)
    print(time.time() - start_time)
    plt.imsave('./outputs/cat-new.jpg', stereogram)

if __name__ == '__main__':
    main()