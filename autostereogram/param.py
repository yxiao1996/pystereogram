class StereogramLineTaskParameter:

    def __init__(self, image_data, rand_data, image_width, rand_width, rand_height, y):
        self.image_data = image_data    # 2-d numpy integer array
        self.rand_data = rand_data      # 2-d numpy integer array
        self.image_width = image_width  # integer
        self.rand_width = rand_width    # integer
        self.rand_height = rand_height  # integer
        self.y = y                      # integer