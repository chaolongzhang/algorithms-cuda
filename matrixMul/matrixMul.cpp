#include <iostream>
#include <windows.h>

using namespace std;

void fillMat(float *mat, size_t rows, size_t cols)
{
    for(int row = 0; row < rows; ++row)
    {
        for(int col = 0; col < cols; ++col)
        {
            mat[row * cols + col] = col;
        }
    }
}

template<int WIDTH>
void matrixMul(float *mat3, float *mat1, float *mat2)
{
    for(int y = 0; y < WIDTH; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            float sum = 0.0f;
            for(int i = 0; i < WIDTH; ++i)
            {
                sum += mat1[y * WIDTH + i] * mat2[i * WIDTH + x];
            }
            mat3[y * WIDTH + x] = sum;
        }
    }
}

template<int WIDTH>
void testMatrixMul()
{
    size_t mem_size = sizeof(float) * WIDTH * WIDTH;
    float *mat1 = (float *)malloc(mem_size);
    float *mat2 = (float *)malloc(mem_size);
    float *mat3 = (float *)malloc(mem_size);

    DWORD start, end;

    // initialize mat1 and mat2
    fillMat(mat1, WIDTH, WIDTH);
    fillMat(mat2, WIDTH, WIDTH);

    start = GetTickCount();
    matrixMul<WIDTH>(mat3, mat1, mat2);
    end = GetTickCount();

    cout << "width: " << WIDTH << "run time: " << (end - start) << " ms" << endl;

    free(mat1);
    free(mat2);
    free(mat3);
}

int main(int argc, char const *argv[])
{
    testMatrixMul<256 >();
    testMatrixMul<512 >();
    testMatrixMul<768 >();
    testMatrixMul<1024>();
    testMatrixMul<1280>();
    testMatrixMul<1536>();
    testMatrixMul<1792>();
    testMatrixMul<2048>();
    return 0;
}