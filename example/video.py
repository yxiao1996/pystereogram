import numpy as np
import matplotlib.image as mpimg
from autostereogram.converter import StereogramConverter
import skvideo.io
from skimage import color
import os
import re
import time

def main():
    image_path = '../rgbd-scenes/bear_front/depth/'
    image_files = os.listdir(image_path)
    frame_converter = StereogramConverter()
    with skvideo.io.FFmpegWriter("outputvideo2.mp4") as video_writer:
        for i in range(1, 296):
            start_time = time.time()
            image_filename = list(filter(re.compile(".*-" + str(i) + ".png").match, image_files))[0]
            image_data = np.array(mpimg.imread(image_path + image_filename) * 255, dtype=int)
            image_data = frame_converter.convert_depth_to_stereogram(image_data)
            output_frame = image_data.astype(np.uint8)
            output_frame = color.grey2rgb(output_frame)
            video_writer.writeFrame(output_frame)
            print(time.time() - start_time)

if __name__ == "__main__":
    main()