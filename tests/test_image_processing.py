import sys
import types

# Stub numpy
numpy = types.ModuleType('numpy')
class FakeArray:
    def __init__(self, shape=(100, 100, 3)):
        self._shape = shape
    @property
    def shape(self):
        return self._shape

def asarray(img):
    return FakeArray()

numpy.asarray = asarray
numpy.array = FakeArray
sys.modules['numpy'] = numpy

# Stub pandas
pandas = types.ModuleType('pandas')
class Row(dict):
    pass
class DataFrame:
    def iterrows(self):
        yield 0, Row({1: 0.1, 2: 0.1, 3: 0.2, 4: 0.2})

def read_csv(*args, **kwargs):
    return DataFrame()

pandas.read_csv = read_csv
sys.modules['pandas'] = pandas

# Stub PIL
PIL = types.ModuleType('PIL')
Image_module = types.ModuleType('Image')
PIL.Image = Image_module
sys.modules['PIL'] = PIL
sys.modules['PIL.Image'] = Image_module

# Stub matplotlib
matplotlib = types.ModuleType('matplotlib')
pyplot = types.ModuleType('pyplot')
patches = types.ModuleType('patches')

class Rectangle:
    def __init__(self, xy, width, height, **kwargs):
        self._x, self._y = xy
        self._width = width
        self._height = height
    def get_x(self):
        return self._x
    def get_y(self):
        return self._y
    def get_width(self):
        return self._width
    def get_height(self):
        return self._height

class Axes:
    def __init__(self):
        self.patches = []
    def imshow(self, pixels):
        pass
    def add_patch(self, patch):
        self.patches.append(patch)

pyplot._last_ax = None

def subplots(*args, **kwargs):
    ax = Axes()
    pyplot._last_ax = ax
    return object(), ax

def show():
    pass

patches.Rectangle = Rectangle
pyplot.subplots = subplots
pyplot.show = show

matplotlib.pyplot = pyplot
matplotlib.patches = patches

sys.modules['matplotlib'] = matplotlib
sys.modules['matplotlib.pyplot'] = pyplot
sys.modules['matplotlib.patches'] = patches

from utility import image_processing

# Patch get_pixels to avoid using PIL
class FakePixels:
    shape = (100, 100, 3)

def fake_get_pixels(_):
    return FakePixels()

image_processing.get_pixels = fake_get_pixels


def test_show_img_with_bounding_box():
    image_processing.show_img_with_bounding_box('dummy.png', 'dummy.txt')
    rect = pyplot._last_ax.patches[0]
    assert rect.get_x() == 10
    assert rect.get_y() == 10
    assert rect.get_width() == 10
    assert rect.get_height() == 10
