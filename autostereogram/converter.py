import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from compute_line import compute_stereogram_line, compute_stereogram_line_with_gil
from skimage.draw import disk

class StereogramLineTaskParameter:

    def __init__(self, image_data, rand_data, image_width, rand_width, rand_height, y):
        self.image_data = image_data    # 2-d numpy integer array
        self.rand_data = rand_data      # 2-d numpy integer array
        self.image_width = image_width  # integer
        self.rand_width = rand_width    # integer
        self.rand_height = rand_height  # integer
        self.y = y                      # integer

def compute_stereogram_line_cython_impl_wrapper(params: StereogramLineTaskParameter):
    return compute_stereogram_line(
        params.image_data,
        params.rand_data,
        params.image_width,
        params.rand_width,
        params.rand_height,
        params.y
    )

# TODO: remove this method after experiment with GIL
def compute_stereogram_line_cython_impl_wrapper_with_gil(params: StereogramLineTaskParameter):
    return compute_stereogram_line_with_gil(
        params.image_data,
        params.rand_data,
        params.image_width,
        params.rand_width,
        params.rand_height,
        params.y
    )

class StereogramConverter:

    def __init__(self, rand_size=64):
        # Create a random seed to build the image from
        self.rand_size = rand_size
        self.rand_data = np.floor(np.random.uniform(0, 255, (rand_size, rand_size))).astype(int)
        [self.rand_height, self.rand_width] = self.rand_data.shape
        self.executor = ThreadPoolExecutor(100)

    def convert_depth_to_stereogram(self, image_data: np.array, draw_helper_dots: bool = False):

        image_data = self._preprocess_image_data(image_data)
        [image_height, image_width] = image_data.shape

        for y in range(image_height):
            line_result = compute_stereogram_line(
                image_data,
                self.rand_data,
                image_width,
                self.rand_width,
                self.rand_height,
                y)
            image_data[y, :] = line_result[:]
        if (draw_helper_dots):
            self._draw_helper_dots(image_data)
        return image_data

    # Convert a depth image stored in a numpy array to auto-stereogram
    def convert_depth_to_stereogram_with_thread_pool(self, image_data: np.array, draw_helper_dots: bool = False):

        image_data = self._preprocess_image_data(image_data)
        [image_height, image_width] = image_data.shape

        # Get the parameters for computing stereogram
        line_params = self._get_task_params(
            image_data,
            self.rand_data,
            image_width,
            image_height,
            self.rand_width,
            self.rand_height
        )

        # Compute the stereogram with thread pool
        line_tasks = {
            self.executor.submit(
                compute_stereogram_line_cython_impl_wrapper,
                param
            ): param for param in line_params
        }
        for future in as_completed(line_tasks):
            param = line_tasks[future]
            line_result = future.result()
            image_data[param.y, :] = line_result[:]
        if (draw_helper_dots):
            self._draw_helper_dots(image_data)
        return image_data

    def _preprocess_image_data(self, image_data: np.array):
        # Scale the pixel values and reduce the depth
        # image_data = image_data - image_data.min()
        # image_data = np.floor(image_data / image_data.max() * self.rand_size / 3)
        image_data = image_data.max() - image_data
        image_data = np.floor(image_data / image_data.max() * self.rand_size / 3)
        image_data = image_data.astype(int)
        return image_data

    def _get_task_params(self, image_data, rand_data, image_width, image_height, rand_width, rand_height):
        return [StereogramLineTaskParameter(
            image_data,
            rand_data,
            image_width,
            rand_width,
            rand_height,
            y
        ) for y in range(image_height)]

    def _draw_helper_dots(self, image):
        [image_height, image_width] = image.shape
        self._draw_circle(image, image_height * 19 / 20, image_width / 2 - self.rand_size / 2)
        self._draw_circle(image, image_height * 19 / 20, image_width / 2 + self.rand_size / 2)

    def _draw_circle(self, image, y, x, r=5):
        [rr, cc] = disk((y, x), r)
        image[rr, cc] = 0

if __name__ == "__main__":
    from skimage import color
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import time
    source_image = mpimg.imread('./example/inputs/cube.jpg')
    if len(source_image.shape) == 3:
        source_image = color.rgb2gray(source_image)
    image_data = np.array(source_image * 255, dtype=int)

    converter = StereogramConverter(128)
    start_time = time.time()
    image_data = image_data.max() - image_data
    image_data = converter.convert_depth_to_stereogram_with_thread_pool(image_data, True).astype(np.uint8)
    print(time.time() - start_time)
    plt.imsave('./example/outputs/cube2.jpg', color.gray2rgb(image_data))