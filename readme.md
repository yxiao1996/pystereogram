## pystereogram: a python library to compute auto-stereogram

pystereogram is a library that can help you compute stereogram from depth image efficiently. 
The performance goal of this library is to support real-time stereogram computation on small devices like raspberry pi.

The algorithm implemented in this package is the abSIRD algorithm, which is then accelerated with cython.

### Usage

```
from autostereogram.converter import StereogramConverter
from skimage import color
import matplotlib.image as mpimg

source_image = mpimg.imread('cube.jpg')
image_data = np.array(source_image * 255, dtype=int)

converter = StereogramConverter()
result = converter.convert_depth_to_stereogram(image_data).astype(np.uint8)
```

### Reference

https://www.mathworks.com/matlabcentral/fileexchange/27649-absird-for-matlab