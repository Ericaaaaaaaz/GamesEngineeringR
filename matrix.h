#pragma once

#include <iostream>
#include <vector>
#include "vec4.h"
#include <immintrin.h>
#include <chrono>

using namespace std;

// Matrix class for 4x4 transformation matrices
class alignas(32) matrix ///////changed: aligns the whole matrix structure to 32 bytes
{

    union
    {
        float m[4][4]; // 2D array representation of the matrix
        float a[16];   // 1D array representation of the matrix for linear access
    };


public:
    // Default constructor initializes the matrix as an identity matrix
    matrix()
    {
        identity();
    }

    void identity()
    {
        memset(m, 0, 16 * sizeof(float));
        a[0] = 1.0f;
        a[5] = 1.0f;
        a[10] = 1.0f;
        a[15] = 1.0f;
    }

    // Access matrix elements by row and column
    float& operator()(unsigned int row, unsigned int col)
    {
        return m[row][col];
    }

    // Display the matrix elements in a readable format
    void display()
    {
        for (unsigned int i = 0; i < 4; i++)
        {
            for (unsigned int j = 0; j < 4; j++)
                std::cout << m[i][j] << '\t';
            std::cout << std::endl;
        }
    }

    // Multiply the matrix by a 4D vector
    // Input Variables:
    // - v: vec4 object to multiply with the matrix
    // Returns the resulting transformed vec4
    vec4 operator * (const vec4& v) const
    {
        vec4 result;
        result[0] = a[0] * v[0] + a[1] * v[1] + a[2] * v[2] + a[3] * v[3];
        result[1] = a[4] * v[0] + a[5] * v[1] + a[6] * v[2] + a[7] * v[3];
        result[2] = a[8] * v[0] + a[9] * v[1] + a[10] * v[2] + a[11] * v[3];
        result[3] = a[12] * v[0] + a[13] * v[1] + a[14] * v[2] + a[15] * v[3];
        return result;
    }
    

    // Multiply the matrix by another matrix
    // Input Variables:
    // - mx: Another matrix to multiply with
    // Returns the resulting matrix
   /*matrix operator * (const matrix& mx) const
   {
        matrix ret;
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                ret.a[row * 4 + col] =
                    a[row * 4 + 0] * mx.a[0 * 4 + col] +
                    a[row * 4 + 1] * mx.a[1 * 4 + col] +
                    a[row * 4 + 2] * mx.a[2 * 4 + col] +
                    a[row * 4 + 3] * mx.a[3 * 4 + col];
            }
        }

        return ret;
   }*/

   matrix operator * (const matrix& mx) const
   {
       matrix ret;

       //row 0
       ret.a[0] = a[0] * mx.a[0] + a[1] * mx.a[4] + a[2] * mx.a[8] + a[3] * mx.a[12];
       ret.a[1] = a[0] * mx.a[1] + a[1] * mx.a[5] + a[2] * mx.a[9] + a[3] * mx.a[13];
       ret.a[2] = a[0] * mx.a[2] + a[1] * mx.a[6] + a[2] * mx.a[10] + a[3] * mx.a[14];
       ret.a[3] = a[0] * mx.a[3] + a[1] * mx.a[7] + a[2] * mx.a[11] + a[3] * mx.a[15];

       //row 1
       ret.a[4] = a[4] * mx.a[0] + a[5] * mx.a[4] + a[6] * mx.a[8] + a[7] * mx.a[12];
       ret.a[5] = a[4] * mx.a[1] + a[5] * mx.a[5] + a[6] * mx.a[9] + a[7] * mx.a[13];
       ret.a[6] = a[4] * mx.a[2] + a[5] * mx.a[6] + a[6] * mx.a[10] + a[7] * mx.a[14];
       ret.a[7] = a[4] * mx.a[3] + a[5] * mx.a[7] + a[6] * mx.a[11] + a[7] * mx.a[15];

       //row 2
       ret.a[8] = a[8] * mx.a[0] + a[9] * mx.a[4] + a[10] * mx.a[8] + a[11] * mx.a[12];
       ret.a[9] = a[8] * mx.a[1] + a[9] * mx.a[5] + a[10] * mx.a[9] + a[11] * mx.a[13];
       ret.a[10] = a[8] * mx.a[2] + a[9] * mx.a[6] + a[10] * mx.a[10] + a[11] * mx.a[14];
       ret.a[11] = a[8] * mx.a[3] + a[9] * mx.a[7] + a[10] * mx.a[11] + a[11] * mx.a[15];

       //row 3
       ret.a[12] = a[12] * mx.a[0] + a[13] * mx.a[4] + a[14] * mx.a[8] + a[15] * mx.a[12];
       ret.a[13] = a[12] * mx.a[1] + a[13] * mx.a[5] + a[14] * mx.a[9] + a[15] * mx.a[13];
       ret.a[14] = a[12] * mx.a[2] + a[13] * mx.a[6] + a[14] * mx.a[10] + a[15] * mx.a[14];
       ret.a[15] = a[12] * mx.a[3] + a[13] * mx.a[7] + a[14] * mx.a[11] + a[15] * mx.a[15];

       return ret;
   }


    // Create a perspective projection matrix
    // Input Variables:
    // - fov: Field of view in radians
    // - aspect: Aspect ratio of the viewport
    // - n: Near clipping plane
    // - f: Far clipping plane
    // Returns the perspective matrix
    static inline matrix makePerspective(float fov, float aspect, float n, float f)
    {
        matrix m;
        //m.zero();
        float tanHalfFov = std::tan(fov / 2.0f);

        m.a[0] = 1.0f / (aspect * tanHalfFov);
        m.a[5] = 1.0f / tanHalfFov;
        m.a[10] = -f / (f - n);
        m.a[11] = -(f * n) / (f - n);
        m.a[14] = -1.0f;
        m.a[15] = 0;
        return m;
    }

    // Create a translation matrix
    // Input Variables:
    // - tx, ty, tz: Translation amounts along the X, Y, and Z axes
    // Returns the translation matrix
    static inline matrix makeTranslation(float tx, float ty, float tz)
    {
        matrix m;
        //m.identity();
        m.a[3] = tx;
        m.a[7] = ty;
        m.a[11] = tz;
        return m;
    }

    // Create a rotation matrix around the Z-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static inline matrix makeRotateZ(float aRad)
    {
        matrix m;
        //m.identity();
        float ct = cos(aRad);
        float st = sin(aRad);
        m.a[0] = ct;
        m.a[1] = -st;
        m.a[4] = st;
        m.a[5] = ct;
        return m;
    }

    // Create a rotation matrix around the X-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static inline matrix makeRotateX(float aRad)
    {
        matrix m;
        //m.identity();
        float ct = cos(aRad);
        float st = sin(aRad);
        m.a[5] = ct;
        m.a[6] = -st;
        m.a[9] = st;
        m.a[10] = ct;
        return m;
    }

    // Create a rotation matrix around the Y-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static inline matrix makeRotateY(float aRad)
    {
        matrix m;
        //m.identity();
        float ct = cos(aRad);
        float st = sin(aRad);
        m.a[0] = ct;
        m.a[2] = st;
        m.a[8] = -st;
        m.a[10] = ct;
        return m;
    }

    // Create a composite rotation matrix from X, Y, and Z rotations
    // Input Variables:
    // - x, y, z: Rotation angles in radians around each axis
    // Returns the composite rotation matrix
    static inline matrix makeRotateXYZ(float x, float y, float z)
    {
        return matrix::makeRotateX(x) * matrix::makeRotateY(y) * matrix::makeRotateZ(z);
    }

    // Create a scaling matrix
    // Input Variables:
    // - s: Scaling factor
    // Returns the scaling matrix
    static inline matrix makeScale(float s)
    {
        matrix m;
        s = max(s, 0.01f); // Ensure scaling factor is not too small
        //m.identity();
        m.a[0] = s;
        m.a[5] = s;
        m.a[10] = s;
        return m;
    }
};


