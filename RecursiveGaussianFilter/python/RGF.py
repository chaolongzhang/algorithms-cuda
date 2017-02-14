# recursive gaussian filter
import numpy as np 

# OpenCV sigma = 0.3(n/2 - 1) + 0.8
def width_to_sigma(width):
    sigma = (width / 2.0 - 1) * 0.3 + 0.8
    return sigma


# YVRG
def calc_coeff(sigma):
    if sigma < 0.5:
        q = 0.11477
    elif sigma < 2.5:
        q = 3.97156 - 4.14554 * np.sqrt(1 - 0.26891 * sigma)
    else:
        q = 0.98711 * sigma + 0.96330
    qq = q ** 2
    qqq = q ** 3
    b0 = 1.57825 + 2.44413 * q + 1.4281 * qq + 0.422205 * qqq
    b1 = 2.44413 * q + 2.85619 * qq + 1.26661 * qqq
    b2 = - (1.4281 * qq + 1.26661 * qqq)
    b3 = 0.422205 * qqq
    B = 1.0 - ((b1 + b2 + b3) / b0)
    return (B, b0, b1, b2, b3)

def forward_pass(in_x, B, b0, b1, b2, b3):
    length = len(in_x)
    out = [0.0 for i in range(length)]
    out[0] = B * in_x[0] + (b1 * in_x[0] + b2 * in_x[0] + b3 * in_x[0]) / b0
    out[1] = B * in_x[1] + (b1 * out[0] + b2 * in_x[0] + b3 * in_x[0]) / b0
    out[2] = B * in_x[2] + (b1 * out[1] + b2 * out[0] + b3 * in_x[0]) / b0
    for i in range(3, length):
        out[i] = B * in_x[i] + (b1 * out[i - 1] + b2 * out[i - 2] + b3 * out[i - 3]) / b0
    return out

def backward_pass(in_x, B, b0, b1, b2, b3):
    length = len(in_x)
    out = [0.0 for i in range(length)]
    out[length - 1] = B * in_x[length - 1] + (b1 * in_x[length - 1] + b2 * in_x[length - 1] + b3 * in_x[length - 1]) / b0
    out[length - 2] = B * in_x[length - 2] + (b1 * out[length - 1] + b2 * in_x[length - 1] + b3 * in_x[length - 1]) / b0
    out[length - 3] = B * in_x[length - 3] + (b1 * out[length - 2] + b2 * out[length - 1] + b3 * in_x[length - 1]) / b0
    for i in range(length - 4, -1, -1):
        out[i] = B * in_x[i] + (b1 * out[i + 1] + b2 * out[i + 2] + b3 * out[i + 3]) / b0
    return out

def yvrg_1d(in_x, sigma = 0.8):
    (B, b0, b1, b2, b3) = calc_coeff(sigma)
    w = forward_pass(in_x, B, b0, b1, b2, b3)
    out = backward_pass(w, B, b0, b1, b2, b3)
    return out

def yvrg_2d(mat, sigma = 0.8):
    w = []
    for x in mat:
        out = yvrg_1d(x, sigma)
        w.append(out)
    w = np.array(w)
    iw = w.transpose()
    out_mat = []
    for y in iw:
        out = yvrg_1d(y, sigma)
        out_mat.append(out)
    out_mat = np.array(out_mat)
    out_mat = out_mat.transpose()
    return out_mat