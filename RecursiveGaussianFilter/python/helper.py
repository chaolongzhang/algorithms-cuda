from scipy import misc
import scipy.io as sciio
from matplotlib import pyplot as plt 

def load_img():
    f = misc.face(gray=True)
    return f

def show_img(img):
    plt.figure()
    plt.gray()
    plt.imshow(img)

def show_all():
    plt.show()

