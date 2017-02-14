/*
* recursive gaussian filter by Young & Van Vliet 
* about recursive gaussian filter, see paper 
* Young I T, Van Vliet L J. Recursive implementation of the Gaussian filter[J]. Signal processing, 1995, 44(2): 139-151.
*/
#ifndef RGF_H
#define RGF_H

float with_to_sigma(float width);

void calc_coeff(float sigma, float &B, float &b0, float &b1, float &b2, float &b3);

/*
out[i] = B * in[i] + (b1 * out[i - 1] + b2 * out[i - 2] + b3 * out[i - 3]) / b0
*/
template<typename T>
void forward_pass(T *out, T *in, size_t len, float B, float b0, float b1, float b2, float b3)
{
    out[0] = B * in[0] + (b1 * in[0] + b2 * in[0] + b3 * in[0]) / b0;
    out[1] = B * in[1] + (b1 * out[0] + b2 * in[0] + b3 * in[0]) / b0;
    out[2] = B * in[2] + (b1 * out[1] + b2 * out[0] + b3 * in[0]) / b0;
    for (int i = 3; i < len; ++i)
    {
        out[i] = B * in[i] + (b1 * out[i - 1] + b2 * out[i - 2] + b3 * out[i - 3]) / b0;
    }
}

/*
out[i] = B * in[i] + (b1 * out[i + 1] + b2 * out[i + 2] + b3 * out[i + 3]) / b0
*/
template<typename T>
void backward_pass(T *out, T *in, size_t len, float B, float b0, float b1, float b2, float b3)
{
    size_t length = len;
    out[length - 1] = B * in[length - 1] + (b1 * in[length - 1] + b2 * in[length - 1] + b3 * in[length - 1]) / b0;
    out[length - 2] = B * in[length - 2] + (b1 * out[length - 1] + b2 * in[length - 1] + b3 * in[length - 1]) / b0;
    out[length - 3] = B * in[length - 3] + (b1 * out[length - 2] + b2 * out[length - 1] + b3 * in[length - 1]) / b0;
    for(int i = len - 4; i >=0; --i)
    {
        out[i] = B * in[i] + (b1 * out[i + 1] + b2 * out[i + 2] + b3 * out[i + 3]) / b0;
    }
}

template<typename T>
void yvrg_1d(T *out, T *in, size_t len, float sigma = 0.8)
{
    float B, b0, b1, b2, b3;
    calc_coeff(sigma, B, b0, b1, b2, b3);

    yvrg_1d(out, in, len, B, bo, b1, b2, b3);
}

template<typename T>
void yvrg_1d(T *out, T *in, size_t len, float B, float b0, float b1, float b2, float b3)
{
    T *w = new T[len];
    forward_pass(w, in, len, B, b0, b1, b2, b3);
    backward_pass(out, w, len, B, b0, b1, b2, b3);

    delete [] w;
}

template<typename T>
void yvrg_2d(T *out, T *in, size_t width, size_t height, float sigma = 0.8)
{
    float B, b0, b1, b2, b3;
    calc_coeff(sigma, B, b0, b1, b2, b3);

    yvrg_2d(out, in, width, height, B, b0, b1, b2, b3);
}

template<typename T>
void yvrg_2d(T *out, T *in, size_t width, size_t height, float B, float b0, float b1, float b2, float b3)
{
    T *w = new T[width * height];

    for(int i = 0; i < height; ++i)
    {
        T *in_offset = in + (i * width);
        T *w_offset = w + (i * width);
        yvrg_1d(w_offset, in_offset, width, B, b0, b1, b2, b3);
    }

    T *temp_in = new T[height];
    T *temp_out = new T[height];
    for(int x = 0; x < width; ++x)
    {
        // copy columh to temp_in
        for(int y = 0; y < height; ++y)
        {
            temp_in[y] = w[y * width + x];
        }
        
        yvrg_1d(temp_out, temp_in, height, B, b0, b1, b2, b3);

        // copy temp_out to out
        for(int y = 0; y < height; ++y)
        {
            out[y * width + x] = temp_out[y];
        }
    }

    delete [] temp_in;
    delete [] temp_out;
    delete [] w;
}

#endif