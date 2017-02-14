import numpy as np 
import helper
import RGF

sigma = 3.5

def main():
    img = helper.load_img()
    
    img2 = RGF.yvrg_2d(img, sigma)
    helper.show_img(img2)
    helper.show_all()

if __name__ == '__main__':
    main()