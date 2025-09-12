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
        __m256 minDepth = _mm256_set1_ps(0.01f);


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
        __m256 lightX = _mm256_broadcast_ss(&L.omega_i[0]);
        __m256 lightY = _mm256_broadcast_ss(&L.omega_i[1]);
        __m256 lightZ = _mm256_broadcast_ss(&L.omega_i[2]);

        //precompute normal components for SIMD
        __m256 n0x = _mm256_broadcast_ss(&v[0].normal[0]);
        __m256 n0y = _mm256_broadcast_ss(&v[0].normal[1]);
        __m256 n0z = _mm256_broadcast_ss(&v[0].normal[2]);
        __m256 n1x = _mm256_broadcast_ss(&v[1].normal[0]);
        __m256 n1y = _mm256_broadcast_ss(&v[1].normal[1]);
        __m256 n1z = _mm256_broadcast_ss(&v[1].normal[2]);
        __m256 n2x = _mm256_broadcast_ss(&v[2].normal[0]);
        __m256 n2y = _mm256_broadcast_ss(&v[2].normal[1]);
        __m256 n2z = _mm256_broadcast_ss(&v[2].normal[2]);


        //lighting constants
        const __m256 kdVec = _mm256_set1_ps(kd);

        const __m256 lightIntensityR = _mm256_set1_ps(L.L[colour::RED]);
        const __m256 lightIntensityG = _mm256_set1_ps(L.L[colour::GREEN]);
        const __m256 lightIntensityB = _mm256_set1_ps(L.L[colour::BLUE]);

        //ambient light, separated by channel
        const __m256 ambientLightR = _mm256_set1_ps(L.ambient[colour::RED]);
        const __m256 ambientLightG = _mm256_set1_ps(L.ambient[colour::GREEN]);
        const __m256 ambientLightB = _mm256_set1_ps(L.ambient[colour::BLUE]);

        //colour components for SIMD
        const __m256 c0r = _mm256_set1_ps(v[0].rgb[colour::RED]);
        const __m256 c0g = _mm256_set1_ps(v[0].rgb[colour::GREEN]);
        const __m256 c0b = _mm256_set1_ps(v[0].rgb[colour::BLUE]);
        const __m256 c1r = _mm256_set1_ps(v[1].rgb[colour::RED]);
        const __m256 c1g = _mm256_set1_ps(v[1].rgb[colour::GREEN]);
        const __m256 c1b = _mm256_set1_ps(v[1].rgb[colour::BLUE]);
        const __m256 c2r = _mm256_set1_ps(v[2].rgb[colour::RED]);
        const __m256 c2g = _mm256_set1_ps(v[2].rgb[colour::GREEN]);
        const __m256 c2b = _mm256_set1_ps(v[2].rgb[colour::BLUE]);

        const __m256 vec255 = _mm256_set1_ps(255.0f);

        //create x offset pattern for 8 pixels
        const __m256 xOffset = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

        for (int y = minY; y < maxY; ++y)
        {
            __m256 py = _mm256_set1_ps((float)y);

            //process 8 pixels at a time
            int x = minX;

            //align to 8-pixel boundary for better performance
            int alignedMinX = (minX + 7) & ~7;

            //handle unaligned start pixels with scalar code
            for (; x < alignedMinX && x < maxX; ++x)
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

            //process 8 pixels at once
            for (; x <= maxX - 8; x += 8)
            {
                //create x coordinates for 8 pixels
                const __m256 px = _mm256_add_ps(_mm256_set1_ps((float)x), xOffset);

                //calculate barycentric coordinates for 8 pixels
                const __m256 alpha = _mm256_mul_ps(getCSIMD(v0x, v0y, v1x, v1y, px, py), invAreaVec);
                const __m256 beta = _mm256_mul_ps(getCSIMD(v1x, v1y, v2x, v2y, px, py), invAreaVec);
                const __m256 gamma = _mm256_sub_ps(_mm256_sub_ps(one, alpha), beta);

                //check if pixels are inside triangle
                const __m256 insideAlpha = _mm256_cmp_ps(alpha, zero, _CMP_GE_OQ);
                const __m256 insideBeta = _mm256_cmp_ps(beta, zero, _CMP_GE_OQ);
                const __m256 insideGamma = _mm256_cmp_ps(gamma, zero, _CMP_GE_OQ);
                const __m256 inside = _mm256_and_ps(insideAlpha, _mm256_and_ps(insideBeta, insideGamma));

                //get mask of which pixels are inside
                const int mask = _mm256_movemask_ps(inside);

                //skip if no pixels are inside
                if (mask == 0) 
                    continue; 

                //interpolate depth for all 8 pixels
                const __m256 depth = _mm256_add_ps(_mm256_mul_ps(beta, v0z),_mm256_add_ps(_mm256_mul_ps(gamma, v1z),_mm256_mul_ps(alpha, v2z)));

                //check depth validity
                const __m256 depthValid = _mm256_cmp_ps(depth, minDepth, _CMP_GT_OQ);

                //load z-buffer values for comparison
                const __m256i v_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                const __m256 zbufferDepths = _mm256_i32gather_ps(&renderer.zbuffer(x, y), v_offsets, sizeof(float));

                //z-buffer test
                const __m256 depthTest = _mm256_cmp_ps(zbufferDepths, depth, _CMP_GT_OQ);

                //combine all tests
                const __m256 shouldRender = _mm256_and_ps(inside, _mm256_and_ps(depthValid, depthTest));
                const int renderMask = _mm256_movemask_ps(shouldRender);

                if (renderMask == 0) continue;

                //vectorized normal interpolation
                __m256 nx = _mm256_add_ps(_mm256_mul_ps(beta, n0x),_mm256_add_ps(_mm256_mul_ps(gamma, n1x),_mm256_mul_ps(alpha, n2x)));
                __m256 ny = _mm256_add_ps(_mm256_mul_ps(beta, n0y),_mm256_add_ps(_mm256_mul_ps(gamma, n1y),_mm256_mul_ps(alpha, n2y)));
                __m256 nz = _mm256_add_ps(_mm256_mul_ps(beta, n0z),_mm256_add_ps(_mm256_mul_ps(gamma, n1z),_mm256_mul_ps(alpha, n2z)));

                //normalize normals (approximate)
                const __m256 invNorm = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(nx, nx),_mm256_add_ps(_mm256_mul_ps(ny, ny),_mm256_mul_ps(nz, nz))));

                nx = _mm256_mul_ps(nx, invNorm);
                ny = _mm256_mul_ps(ny, invNorm);
                nz = _mm256_mul_ps(nz, invNorm);

                //compute dot product with light
                __m256 dotProduct = _mm256_add_ps(_mm256_mul_ps(lightX, nx),_mm256_add_ps(_mm256_mul_ps(lightY, ny),_mm256_mul_ps(lightZ, nz)));
                dotProduct = _mm256_max_ps(dotProduct, zero);

                //interpolate colours
                __m256 cr = _mm256_add_ps(_mm256_mul_ps(beta, c0r),_mm256_add_ps(_mm256_mul_ps(gamma, c1r),_mm256_mul_ps(alpha, c2r)));
                __m256 cg = _mm256_add_ps(_mm256_mul_ps(beta, c0g),_mm256_add_ps(_mm256_mul_ps(gamma, c1g),_mm256_mul_ps(alpha, c2g)));
                __m256 cb = _mm256_add_ps(_mm256_mul_ps(beta, c0b),_mm256_add_ps(_mm256_mul_ps(gamma, c1b),_mm256_mul_ps(alpha, c2b)));

                const __m256 diffuseTermR = _mm256_mul_ps(lightIntensityR, dotProduct);
                const __m256 diffuseTermG = _mm256_mul_ps(lightIntensityG, dotProduct);
                const __m256 diffuseTermB = _mm256_mul_ps(lightIntensityB, dotProduct);

                const __m256 ambientTermR = _mm256_mul_ps(ambientLightR, kdVec);
                const __m256 ambientTermG = _mm256_mul_ps(ambientLightG, kdVec);
                const __m256 ambientTermB = _mm256_mul_ps(ambientLightB, kdVec);

                const __m256 lightContributionR = _mm256_add_ps(diffuseTermR, ambientTermR);
                const __m256 lightContributionG = _mm256_add_ps(diffuseTermG, ambientTermG);
                const __m256 lightContributionB = _mm256_add_ps(diffuseTermB, ambientTermB);

                const __m256 finalFactorR = _mm256_mul_ps(kdVec, lightContributionR);
                const __m256 finalFactorG = _mm256_mul_ps(kdVec, lightContributionG);
                const __m256 finalFactorB = _mm256_mul_ps(kdVec, lightContributionB);

                cr = _mm256_mul_ps(cr, finalFactorR);
                cg = _mm256_mul_ps(cg, finalFactorG);
                cb = _mm256_mul_ps(cb, finalFactorB);

                cr = _mm256_min_ps(_mm256_max_ps(cr, zero), one);
                cg = _mm256_min_ps(_mm256_max_ps(cg, zero), one);
                cb = _mm256_min_ps(_mm256_max_ps(cb, zero), one);

                cr = _mm256_mul_ps(cr, vec255);
                cg = _mm256_mul_ps(cg, vec255);
                cb = _mm256_mul_ps(cb, vec255);

                const __m256i writeMask = _mm256_castps_si256(shouldRender);

                _mm256_maskstore_ps(&renderer.zbuffer(x, y), writeMask, depth);

                if (renderMask != 0)
                {
                    //store the calculated colours into temporary arrays
                    float rArray[8], gArray[8], bArray[8];
                    _mm256_storeu_ps(rArray, cr);
                    _mm256_storeu_ps(gArray, cg);
                    _mm256_storeu_ps(bArray, cb);

                    //loop through the 8 pixels and draw only the ones that passed
                    for (int i = 0; i < 8; ++i)
                    {
                        if (renderMask & (1 << i))
                        {
                            renderer.canvas.draw(x + i, y,
                                (unsigned char)rArray[i],
                                (unsigned char)gArray[i],
                                (unsigned char)bArray[i]);
                        }
                    }
                }

            }

            //handle remaining pixels
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
