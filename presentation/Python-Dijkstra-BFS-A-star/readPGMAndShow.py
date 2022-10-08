import re
import numpy as np
from PIL import Image
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == "__main__":
    from matplotlib import pyplot
    file_path = "/Users/wataruoshima/CSCI651/presentation/Python-Dijkstra-BFS-A-star/files/map_sh.pgm"
    file = "/Users/wataruoshima/CSCI651/presentation/Python-Dijkstra-BFS-A-star/img/1.png"
    image = Image.open(file_path)
    print(image.format)
    print(image.size)
    print(image.mode)
    np_img = np.array(image)
    print(np_img)
    for i in np_img:
        for j in i:
            if j != 205:
                print(j)

    print(np_img.shape)
    print(np_img)
    numpydata = np.asarray(image)
    # <class 'numpy.ndarray'>
    print(type(numpydata))
    
    #  shape
    print(numpydata.shape)

    # image = read_pgm(image, byteorder='<')
    # pyplot.imshow(image, pyplot.cm.gray)
    # pyplot.show()