from timeit import default_timer as timer

def matrixMul(mat1, mat2, width):
    mat3 = [ [ 0 for x in range(width) ] for y in range(width) ]
    for y in range(width):
        for x in range(width):
            temp = 0
            for i in range(width):
                temp += mat1[y][i] * mat2[i][x]
            mat3[y][x] = temp
    return mat3


def testMul(mat1, mat2, width):
    start = timer()
    matrixMul(mat1, mat2, width)
    end = timer()
    print("%d matrix: %f ms" % (width, (end - start) * 1000))

def main():
    for width in range(256, 2048 + 1, 256):
        mat1 = [ [ x * 1.0 for x in range(width) ] for y in range(width) ]
        mat2 = [ [ 2.5 for x in range(width) ] for y in range(width) ]
        testMul(mat1, mat2, width)

if __name__ == '__main__':
    main()
