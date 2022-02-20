from skimage.draw import disk
import numpy as np


class SirdsConverter():

    def convert_depth_to_stereogram_with_sird(
            self,
            image_data: np.array,
            draw_helper_dots: bool = False,
            depth_of_field: float = 1 / 3.0,
            dot_per_inch = 72
    ):
        """
        Use the original SIRDS algorithm to compute stereogram from depth image.
        paper: https://www.cs.waikato.ac.nz/~ihw/papers/94-HWT-SI-IHW-SIRDS-paper.pdf
        TODO: cythonize this code to make it faster.
        :param depth_of_field: a value between 0 and 1 to adjust the maximum depth of the stereogram. Depth goes to max as value approach 1.
        :param dot_per_inch: DPI used to compute the stereogram.
        :param image_data: the input depth image to compute stereogram.
        :param draw_helper_dots: a boolean flag to decide whether to draw the helper black dots.
        :return: the derived stereogram.
        """
        DPI = dot_per_inch
        mu = depth_of_field
        E = np.round(2.5 * DPI)

        def separation(z):
            return np.round((1 - mu * z) * E / (2 - mu * z))

        far = separation(0)
        # Scale the pixel values to between 0 and 1
        image_data = image_data.max() - image_data
        image_data = image_data - image_data.min()
        image_data = image_data / image_data.max()
        image_data = image_data.astype(float)
        [image_height, image_width] = image_data.shape
        z = image_data
        stereogram = np.zeros((image_height, image_width))
        for y in range(image_height):
            pix = np.zeros(image_width)
            same = np.zeros(image_width, dtype=int)

            for x in range(image_width):
                same[x] = x

            for x in range(image_width):
                s = separation(z[y, x])
                left = int(x - s / 2)
                right = int(left + s)
                if (0 <= left and right < image_width):
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
                    pix[x] = np.floor(np.random.uniform(0, 255))
                else:
                    pix[x] = pix[int(same[x])]
            stereogram[y, :] = pix

        if (draw_helper_dots):
            self._draw_circle(stereogram, image_height * 19 / 20, image_width / 2 - far / 2)
            self._draw_circle(stereogram, image_height * 19 / 20, image_width / 2 + far / 2)
        return stereogram

    def _draw_circle(self, image, y, x, r=5):
        [rr, cc] = disk((y, x), r)
        image[rr, cc] = 0