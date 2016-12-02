import numpy as np 
from timeit import default_timer as timer

def matrixMul(matrix1, matrix2):
    return matrix1 * matrix2
    '''
    (H1, W1) = matrix1.shape
    (H2, W2) = matrix2.shape

    matrix3 = np.matrix(np.zeros((H1, W2), dtype=np.float32))
    for y in range(H1):
        for x in range(W2):
            temp = 0
            for i in range(W1):
                temp += matrix1[y, i] * matrix2[i, x]
            matrix3[y, x] = temp
    return matrix3
    '''


def testMul(matrix1, matrix2, width):
    start = timer()
    matrixMul(matrix1, matrix2)
    end = timer()
    print("%d matrix: %f ms" % (width, (end - start) * 1000))

def main():
    for width in range(256, 2048 + 1, 256):
        matrix1 = np.matrix([ [ x * 1.0 for x in range(width) ] for y in range(width) ], dtype=np.float32)
        matrix2 = np.matrix([ [ 2.5 for x in range(width) ] for y in range(width) ], dtype=np.float32)
        testMul(matrix1, matrix2, width)

if __name__ == '__main__':
    main()
