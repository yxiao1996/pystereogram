## pystereogram: a python library to compute auto-stereogram

This library is used to support my website [Stereogram Maker](https://yx.stereogram-maker.com/).

### Introduction

pystereogram is a library that can help you compute stereogram from depth image efficiently. 
The performance goal of this library is to support real-time stereogram computation on small devices like raspberry pi.

The algorithm implemented in this package is the abSIRD algorithm, which is then accelerated with cython.

### Usage

#### Basic Usage: Create stereogram with greyscale white noise

```
from autostereogram.converter import StereogramConverter
from skimage import color
import matplotlib.image as mpimg

source_image = mpimg.imread('cube.jpg')

# Use numpy to randomly generate some noise
image_data = np.array(source_image * 255, dtype=int)

converter = StereogramConverter()
result = converter.convert_depth_to_stereogram(image_data).astype(np.uint8)
```

#### Create stereogram with RGB noise patch

This library also has an method to generate stereogram with RGB noise patch.

```
stereogram = stereogram_converter.convert_depth_to_stereogram_with_rgb_texture(
            depth_map,
            colored_texture,
            draw_helper_dots=True
        )
```
### Reference

https://www.mathworks.com/matlabcentral/fileexchange/27649-absird-for-matlab

https://www.cs.waikato.ac.nz/~ihw/papers/94-HWT-SI-IHW-SIRDS-paper.pdf
