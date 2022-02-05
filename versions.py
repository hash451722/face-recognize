import platform

import cv2
import numpy as np
import matplotlib

def print_version():
    print("Python", platform.python_version())
    print("OpenCV", cv2.__version__)
    print("Numpy", np.__version__)
    print("Matplotlib", matplotlib.__version__)
        

if __name__ == '__main__':
    print_version()
