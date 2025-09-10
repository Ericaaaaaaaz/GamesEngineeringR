#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <immintrin.h>

using namespace std;

// Simple support class for a 2D vector
class vec2D
{
public:
    float x, y;

    // Default constructor initializes both components to 0
    vec2D() : x(0.f), y(0.f)
    {

    };

    // Constructor initializes components with given values
    vec2D(float _x, float _y) : x(_x), y(_y)
    {

    }

    // Constructor initializes components from a vec4
    vec2D(const vec4& v) : x(v[0]), y(v[1])
    {

    }

    // Display the vector components
    inline void display() const
    {
        std::cout << x << '\t' << y << std::endl;
    }

    // Overloaded subtraction operator for vector subtraction
    inline vec2D operator- (vec2D& v)
    {
        return vec2D(x - v.x, y - v.y);
    }
};

// Class representing a triangle for rendering purposes
class triangle
{
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    colour col[3];     // Colors for each vertex of the triangle
    vec2D e1, e2;
    float invArea;


public:
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3)
        : v{ v1, v2, v3 }, e1(v2.p - v1.p), e2(v3.p - v1.p)
    {
        /*v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        e1 = v[1].p - v[0].p;
        e2 = v[2].p - v[0].p;*/

        // Calculate the 2D area of the triangle
        area = abs(e1.x * e2.y - e1.y * e2.x);
        invArea = (area > 0.f) ? 1.f / area : 0.f;
    }

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    inline float getC(vec2D v1, vec2D v2, vec2D p)
    {
        return (p.y - v1.y) * (v2.x - v1.x) - (p.x - v1.x) * (v2.y - v1.y);
    }

    inline __m256 getCSIMD(__m256 v1x, __m256 v1y, __m256 v2x, __m256 v2y, __m256 px, __m256 py)
    {
        __m256 dy = _mm256_sub_ps(py, v1y);
        __m256 dx21 = _mm256_sub_ps(v2x, v1x);
        __m256 dx = _mm256_sub_ps(px, v1x);
        __m256 dy21 = _mm256_sub_ps(v2y, v1y);

        return _mm256_sub_ps(_mm256_mul_ps(dy, dx21), _mm256_mul_ps(dx, dy21));
    }

    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma)
    {

        alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) * invArea;
        beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) * invArea;
        gamma = 1.0f - alpha - beta;

        return (alpha >= 0.f && beta >= 0.f && gamma >= 0.f);

    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    inline T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3)
    {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }

    

    void drawSIMD(Renderer& renderer, Light& L, float ka, float kd)
    {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);

        if (invArea == 0.f)
            return;

        int minX = (int)minV.x, maxX = (int)ceil(maxV.x);
        int minY = (int)minV.y, maxY = (int)ceil(maxV.y);

        L.omega_i.normalise();

        //prepare constant values for SIMD
        __m256 invAreaVec = _mm256_broadcast_ss(&invArea);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 zero = _mm256_set1_ps(0.0f);

        //vertex positions for barycentric calculations
        __m256 v0x = _mm256_broadcast_ss(&v[0].p[0]);
        __m256 v0y = _mm256_broadcast_ss(&v[0].p[1]);
        __m256 v1x = _mm256_broadcast_ss(&v[1].p[0]);
        __m256 v1y = _mm256_broadcast_ss(&v[1].p[1]);
        __m256 v2x = _mm256_broadcast_ss(&v[2].p[0]);
        __m256 v2y = _mm256_broadcast_ss(&v[2].p[1]);

        //vertex attributes for interpolation
        __m256 v0z = _mm256_broadcast_ss(&v[0].p[2]);
        __m256 v1z = _mm256_broadcast_ss(&v[1].p[2]);
        __m256 v2z = _mm256_broadcast_ss(&v[2].p[2]);

        //light direction
        __m256 light_x = _mm256_broadcast_ss(&L.omega_i[0]);
        __m256 light_y = _mm256_broadcast_ss(&L.omega_i[1]);
        __m256 light_z = _mm256_broadcast_ss(&L.omega_i[2]);

        const __m256 lightIntensityR = _mm256_broadcast_ss(&L.L[colour::RED]);
        const __m256 lightIntensityG = _mm256_broadcast_ss(&L.L[colour::GREEN]);
        const __m256 lightIntensityB = _mm256_broadcast_ss(&L.L[colour::BLUE]);

        //ambient light, separated by channel
        const __m256 ambientLightR = _mm256_broadcast_ss(&L.ambient[colour::RED]);
        const __m256 ambientLightG = _mm256_broadcast_ss(&L.ambient[colour::GREEN]);
        const __m256 ambientLightB = _mm256_broadcast_ss(&L.ambient[colour::BLUE]);
        //lighting constants
        __m256 kd_vec = _mm256_set1_ps(kd);

        for (int y = minY; y < maxY; ++y)
        {
            __m256 py = _mm256_set1_ps((float)y);

            // Process 8 pixels at a time
            int x = minX;
            for (; x <= maxX - 8; x += 8)
            {
                //create x coordinates for 8 pixels
                __m256 px = _mm256_set_ps(x + 7.0f, x + 6.0f, x + 5.0f, x + 4.0f,
                    x + 3.0f, x + 2.0f, x + 1.0f, x + 0.0f);

                //calculate barycentric coordinates for 8 pixels
                __m256 alpha = _mm256_mul_ps(getCSIMD(v0x, v0y, v1x, v1y, px, py), invAreaVec);
                __m256 beta = _mm256_mul_ps(getCSIMD(v1x, v1y, v2x, v2y, px, py), invAreaVec);
                __m256 gamma = _mm256_sub_ps(_mm256_sub_ps(one, alpha), beta);

                //check if pixels are inside triangle
                __m256 inside = _mm256_and_ps(_mm256_cmp_ps(alpha, zero, _CMP_GE_OQ),
                    _mm256_and_ps(_mm256_cmp_ps(beta, zero, _CMP_GE_OQ),
                        _mm256_cmp_ps(gamma, zero, _CMP_GE_OQ)));

                //skip if no pixels are inside
                int mask = _mm256_movemask_ps(inside);
                if (mask == 0) continue;

                //interpolate depth
                __m256 depth = _mm256_add_ps(_mm256_mul_ps(beta, v0z),
                    _mm256_add_ps(_mm256_mul_ps(gamma, v1z),
                        _mm256_mul_ps(alpha, v2z)));

                //process each pixel that passed the test
                float depths[8], alphas[8], betas[8], gammas[8];
                _mm256_storeu_ps(depths, depth);
                _mm256_storeu_ps(alphas, alpha);
                _mm256_storeu_ps(betas, beta);
                _mm256_storeu_ps(gammas, gamma);

                for (int i = 0; i < 8; i++)
                {
                    if (mask & (1 << i))
                    {
                        int px_coord = x + i;

                        // Z-buffer test
                        if (renderer.zbuffer(px_coord, y) > depths[i] && depths[i] > 0.01f)
                        {
                            // Interpolate normal (not vectorized for simplicity)
                            vec4 normal = interpolate(betas[i], gammas[i], alphas[i],
                                v[0].normal, v[1].normal, v[2].normal);
                            normal.normalise();

                            // Shading calculation
                            float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
                            colour c = interpolate(betas[i], gammas[i], alphas[i],
                                v[0].rgb, v[1].rgb, v[2].rgb);
                            c.clampColour();

                            colour a = (c * kd) * (L.L * dot + (L.ambient * kd));

                            unsigned char r, g, b;
                            a.toRGB(r, g, b);

                            renderer.canvas.draw(px_coord, y, r, g, b);
                            renderer.zbuffer(px_coord, y) = depths[i];
                        }
                    }
                }
            }

            // Handle remaining pixels (less than 8)
            for (; x < maxX; ++x)
            {
                float alpha, beta, gamma;
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma))
                {
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);

                    if (renderer.zbuffer(x, y) > depth && depth > 0.01f)
                    {
                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                        normal.normalise();

                        float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();

                        colour a = (c * kd) * (L.L * dot + (L.ambient * kd));

                        unsigned char r, g, b;
                        a.toRGB(r, g, b);

                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }
    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
    void draw(Renderer& renderer, Light& L, float ka, float kd)
    {
        vec2D minV, maxV;

        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (invArea == 0.f) return;


        int minX = (int)minV.x, maxX = (int)ceil(maxV.x);
        int minY = (int)minV.y, maxY = (int)ceil(maxV.y);

        L.omega_i.normalise();


        // Iterate over the bounding box and check each pixel
        for (int y = minY; y < maxY; ++y)
        {
            for (int x = minX; x < maxX; ++x)
            {
                float alpha, beta, gamma;

                // Check if the pixel lies inside the triangle
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma))
                {

                    // Interpolate color, depth, and normals
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);

                    // Perform Z-buffer test and apply shading
                    if (renderer.zbuffer(x, y) > depth && depth > 0.01f)
                    {
                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                        normal.normalise();

                        // typical shader begin

                        float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();

                        colour a = (c * kd) * (L.L * dot + (L.ambient * kd));
                        // typical shader end
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);


                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }

    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    void getBounds(vec2D& minV, vec2D& maxV) const
    {
        minV = maxV = vec2D(v[0].p);

        for (unsigned int i = 1; i < 3; i++)
        {
            minV.x = min(minV.x, v[i].p[0]);
            minV.y = min(minV.y, v[i].p[1]);
            maxV.x = max(maxV.x, v[i].p[0]);
            maxV.y = max(maxV.y, v[i].p[1]);
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) const
    {

        getBounds(minV, maxV);
        minV.x = max(minV.x, 0.f);
        minV.y = max(minV.y, 0.f);
        maxV.x = min(maxV.x, static_cast<float>(canvas.getWidth()));
        maxV.y = min(maxV.y, static_cast<float>(canvas.getHeight()));
        /*maxV.x = min(maxV.x, canvas.getWidth());
        maxV.y = min(maxV.y, canvas.getHeight());*/
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas)
    {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        int minX = (int)minV.x, maxX = (int)ceil(maxV.x);
        int minY = (int)minV.y, maxY = (int)ceil(maxV.y);

        for (int y = minY; y < maxY; ++y)
        {
            for (int x = minX; x < maxX; ++x)
            {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() 
    {
        for (auto& vertex : v)
        {
            vertex.p.display();
        }
        std::cout << std::endl;
    }
};
