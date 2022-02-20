import numpy as np
cimport numpy as np
DTYPE = np.int64
ctypedef np.int_t DTYPE_t
from cython.parallel import prange

def compute_stereogram_line(np.ndarray image_data,
                            np.ndarray rand_data,
                            int image_width,
                            int rand_width,
                            int rand_height,
                            int y):

    # assert image_data.dtype == DTYPE
    # assert rand_data.dtype == DTYPE

    cdef int x_index, x
    cdef int y_index = np.mod(y, rand_height - 1)
    cdef long[:, :] image_data_view = image_data
    cdef long[:, :] rand_data_view = rand_data
    cdef long[:] result = np.zeros((image_width), dtype=image_data.dtype)
    with nogil:
        # Start from the right most copt of the random seed
        for x in range(image_width - 1, image_width - rand_width - 1, -1):
            x_index = (x - image_data_view[y, x]) % (rand_width - 1)
            result[x] = rand_data_view[y_index, x_index]

        # Recalculate all the remaining pixels on the line
        for x in range(image_width - rand_width - 1, 0, -1):
            x_index = x + rand_width - image_data_view[y, x]
            result[x] = result[x_index]
    return np.array(result)

# TODO: remove this method after experiment with GIL
def compute_stereogram_line_with_gil(np.ndarray image_data,
                            np.ndarray rand_data,
                            int image_width,
                            int rand_width,
                            int rand_height,
                            int y):

    assert image_data.dtype == DTYPE
    assert rand_data.dtype == DTYPE

    cdef int x_index, x
    cdef int y_index = np.mod(y, rand_height - 1)
    cdef long[:, :] image_data_view = image_data
    cdef long[:, :] rand_data_view = rand_data
    cdef long[:] result = np.zeros((image_width), dtype=DTYPE)
    # Start from the right most copt of the random seed
    for x in range(image_width - 1, image_width - rand_width - 1, -1):
        x_index = (x - image_data_view[y, x]) % (rand_width - 1)
        result[x] = rand_data_view[y_index, x_index]

    # Recalculate all the remaining pixels on the line
    for x in range(image_width - rand_width - 1, 0, -1):
        x_index = x + rand_width - image_data_view[y, x]
        result[x] = result[x_index]
    return np.array(result)


def compute_stereogram(np.ndarray image_data,
                            np.ndarray rand_data,
                            int image_width,
                            int image_height,
                            int rand_width,
                            int rand_height):
    cdef int y
    cdef int y_index
    cdef int x_index, x
    cdef int image_width_c = image_width
    cdef int rand_width_c = rand_width
    cdef int rand_height_c = rand_height
    cdef long[:, :] image_data_view = image_data
    cdef long[:, :] rand_data_view = rand_data
    cdef long[:, :] result = np.zeros((image_width, image_height), dtype=image_data.dtype)

    for y in prange(image_height, nogil=True):
        y_index = np.mod(y, rand_height - 1)

        with nogil:
            # Start from the right most copt of the random seed
            for x in range(image_width - 1, image_width - rand_width - 1, -1):
                x_index = (x - image_data_view[y, x]) % (rand_width - 1)
                result[y, x] = rand_data_view[y_index, x_index]

            # Recalculate all the remaining pixels on the line
            for x in range(image_width - rand_width - 1, 0, -1):
                x_index = x + rand_width - image_data_view[y, x]
                result[y, x] = result[y, x_index]
    return np.array(result)