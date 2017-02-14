
#include <cmath>
#include "RGF.h"

using namespace std;

/*
OpenCV sigma = 0.3(n/2 - 1) + 0.8
*/
float with_to_sigma(float width)
{
    float sigma = (width / 2.0 - 1.0) * 0.3 + 0.8;
    return sigma;
}

void calc_coeff(float sigma, float &B, float &b0, float &b1, float &b2, float &b3)
{
    float q = 0.0F;
    if(sigma < 0.5)
    {
        q = 0.11477;
    }
    else if (sigma < 2.5)
    {
        q = 3.97156 - 4.14554 * sqrt(1 - 0.26891 * sigma);
    }
    else
    {
        q = 0.98711 * sigma + 0.96330;
    }
    float qq = q * q;
    float qqq = q * q * q;
    b0 = 1.57825 + 2.44413 * q + 1.4281 * qq + 0.422205 * qqq;
    b1 = 2.44413 * q + 2.85619 * qq + 1.26661 * qqq;
    b2 = - (1.4281 * qq + 1.26661 * qqq);
    b3 = 0.422205 * qqq;
    B = 1.0 - ((b1 + b2 + b3) / b0);
}

