#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <immintrin.h>
#include <thread> 
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>

using namespace std;

namespace tri_internal 
{

    class RowPool 
    {
        std::mutex submit_m_;
        std::atomic<int> tickets_{ 0 };

    public:
        RowPool() 
        {
            unsigned int hw = max(1u, std::thread::hardware_concurrency());
            max_threads_ = min(11u, hw ? hw : 1u);  
            start_workers(max_threads_);
        }

        ~RowPool() 
        {
            {
                std::lock_guard<std::mutex> lk(m_);
                quitting_ = true;
                job_ready_ = true;
            }
            cv_.notify_all();
            for (auto& t : workers_) if (t.joinable()) t.join();
        }

        template <class F>
        void parallel_rows(int globalY0, int globalY1, int rowsPerTask, F f, unsigned int useThreads = 0)
        {
            std::unique_lock<std::mutex> submit_lock(submit_m_);
            if (globalY0 >= globalY1) return;

            const int H = globalY1 - globalY0;
            job_chunk_ = max(1, rowsPerTask);
            const int numTasks = (H + job_chunk_ - 1) / job_chunk_;

            const unsigned int T = clamp_threads(useThreads);
            const unsigned int active = std::min<unsigned int>(T, max(1, numTasks));

            if (active <= 1) 
            {
                for (int y = globalY0; y < globalY1; y += job_chunk_)
                    f(y, min(globalY1, y + job_chunk_));
                return;
            }

            {
                std::unique_lock<std::mutex> lk(m_);
                next_row_.store(globalY0, std::memory_order_relaxed);
                job_y0_ = globalY0;
                job_y1_ = globalY1;
                job_chunk_ = max(1, rowsPerTask);
                job_ = [this, f]() 
                    {
                        for (;;) 
                        {
                            int y0 = next_row_.fetch_add(job_chunk_, std::memory_order_relaxed);
                            if (y0 >= job_y1_) break;
                            int y1 = min(job_y1_, y0 + job_chunk_);
                            f(y0, y1);
                        }
                    };
          
                pending_workers_ = active;                                
                tickets_.store((int)active, std::memory_order_relaxed); 
                job_ready_ = true;
            }

            //wake workers
            cv_.notify_one();

            for (;;) 
            {
                int y0 = next_row_.fetch_add(job_chunk_, memory_order_relaxed);
                if (y0 >= job_y1_) break;
                int y1 = min(job_y1_, y0 + job_chunk_);
                f(y0, y1);
            }

            //wait until all T workers consumed the job
            unique_lock<mutex> lk(m_);
            done_cv_.wait(lk, [&] 
                { 
                    return pending_workers_ == 0; 
                });
        }

        static RowPool& instance() 
        {
            static RowPool pool;
            return pool;
        }

    private:
        
        unsigned int clamp_threads(unsigned int req) const 
        {
            if (req == 0) 
                return max_threads_;
            return min(req, max_threads_);
        }

        void start_workers(unsigned int T) 
        {
            workers_.reserve(T);
            for (unsigned int i = 0; i < T; ++i) 
            {
                workers_.emplace_back([this] 
                {
                    unique_lock<mutex> lk(m_);
                    for (;;) 
                    {
                        
                        cv_.wait(lk, [&] 
                            {
                                return quitting_ || (job_ready_ && tickets_.load(std::memory_order_relaxed) > 0);
                            });

                        if (quitting_) 
                            return;

                        tickets_.fetch_sub(1, std::memory_order_acq_rel);

                        // Copy job and run without holding the lock
                        auto job = job_;
                        lk.unlock();

                        if (job) 
                            job();

                        lk.lock();

                        if (--pending_workers_ == 0) 
                        {
                            job_ready_ = false;           // close the job
                            done_cv_.notify_one();        // wake submitter waiting in parallel_rows()
                        }
                    }
                });
            }
        }
     

        vector<thread> workers_;
        unsigned int max_threads_ = 1;

        mutex m_;
        condition_variable cv_;
        condition_variable done_cv_;
        bool job_ready_ = false;
        bool quitting_ = false;
        unsigned int pending_workers_ = 0;

        function<void()> job_;
        atomic<int> next_row_{ 0 };
        int job_y0_ = 0, job_y1_ = 0, job_chunk_ = 1;
    };

} 


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

struct TriSIMDPre 
{
    int minX, maxX;
    vec4 omegaN;
    __m256 invAreaVec, one, zero, minDepth;
    __m256 v0x, v0y, v1x, v1y, v2x, v2y;
    __m256 v0z, v1z, v2z;
    __m256 lightX, lightY, lightZ;
    __m256 n0x, n0y, n0z, n1x, n1y, n1z, n2x, n2y, n2z;
    __m256 kdVec;
    __m256 lightIntensityR, lightIntensityG, lightIntensityB, ambientLightR, ambientLightG, ambientLightB;
    __m256 c0r, c0g, c0b, c1r, c1g, c1b, c2r, c2g, c2b;
    __m256 vec255, xOffset;
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

    inline __m256 getCoordinatesSIMD(__m256 v1x, __m256 v1y, __m256 v2x, __m256 v2y, __m256 px, __m256 py)
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

    void buildPre(Light& L, float kd, TriSIMDPre& P) 
    {
        P.omegaN = L.omega_i; 
        P.omegaN.normalise();

        //prepare constant values for SIMD
        P.invAreaVec = _mm256_broadcast_ss(&invArea);
        P.one = _mm256_set1_ps(1.0f);
        P.zero = _mm256_set1_ps(0.0f);
        P.minDepth = _mm256_set1_ps(0.01f);

        //vertex positions for barycentric calculations
        P.v0x = _mm256_broadcast_ss(&v[0].p[0]); 
        P.v0y = _mm256_broadcast_ss(&v[0].p[1]);
        P.v1x = _mm256_broadcast_ss(&v[1].p[0]); 
        P.v1y = _mm256_broadcast_ss(&v[1].p[1]);
        P.v2x = _mm256_broadcast_ss(&v[2].p[0]); 
        P.v2y = _mm256_broadcast_ss(&v[2].p[1]);

        //vertex attributes for interpolation
        P.v0z = _mm256_broadcast_ss(&v[0].p[2]);
        P.v1z = _mm256_broadcast_ss(&v[1].p[2]);
        P.v2z = _mm256_broadcast_ss(&v[2].p[2]);

        //light direction
        P.lightX = _mm256_set1_ps(P.omegaN[0]);
        P.lightY = _mm256_set1_ps(P.omegaN[1]);
        P.lightZ = _mm256_set1_ps(P.omegaN[2]);

        //precompute normal components for SIMD
        P.n0x = _mm256_broadcast_ss(&v[0].normal[0]);
        P.n0y = _mm256_broadcast_ss(&v[0].normal[1]); 
        P.n0z = _mm256_broadcast_ss(&v[0].normal[2]);
        P.n1x = _mm256_broadcast_ss(&v[1].normal[0]); 
        P.n1y = _mm256_broadcast_ss(&v[1].normal[1]); 
        P.n1z = _mm256_broadcast_ss(&v[1].normal[2]);
        P.n2x = _mm256_broadcast_ss(&v[2].normal[0]); 
        P.n2y = _mm256_broadcast_ss(&v[2].normal[1]); 
        P.n2z = _mm256_broadcast_ss(&v[2].normal[2]);

        //lighting constants
        P.kdVec = _mm256_set1_ps(kd);

        P.lightIntensityR = _mm256_set1_ps(L.L[colour::RED]);
        P.lightIntensityG = _mm256_set1_ps(L.L[colour::GREEN]);
        P.lightIntensityB = _mm256_set1_ps(L.L[colour::BLUE]);
        P.ambientLightR = _mm256_set1_ps(L.ambient[colour::RED]);
        P.ambientLightG = _mm256_set1_ps(L.ambient[colour::GREEN]);
        P.ambientLightB = _mm256_set1_ps(L.ambient[colour::BLUE]);

        P.c0r = _mm256_set1_ps(v[0].rgb[colour::RED]);
        P.c0g = _mm256_set1_ps(v[0].rgb[colour::GREEN]);
        P.c0b = _mm256_set1_ps(v[0].rgb[colour::BLUE]);
        P.c1r = _mm256_set1_ps(v[1].rgb[colour::RED]);
        P.c1g = _mm256_set1_ps(v[1].rgb[colour::GREEN]);
        P.c1b = _mm256_set1_ps(v[1].rgb[colour::BLUE]);
        P.c2r = _mm256_set1_ps(v[2].rgb[colour::RED]);
        P.c2g = _mm256_set1_ps(v[2].rgb[colour::GREEN]);
        P.c2b = _mm256_set1_ps(v[2].rgb[colour::BLUE]);

        P.vec255 = _mm256_set1_ps(255.0f);

        P.xOffset = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    }

    void drawSIMDrows(Renderer& renderer, const TriSIMDPre& P,Light& L, float ka, float kd, int yStart, int yEnd)
    {
        if (invArea == 0.f)
            return;

        int minX = P.minX;
        int maxX = P.maxX;
        int minY = yStart;
        int maxY = yEnd;

        if (minY >= maxY) 
            return;

        const __m256 invAreaVec = P.invAreaVec;
        const __m256 one = P.one;
        const __m256 zero = P.zero;
        const __m256 minDepth = P.minDepth;

        const __m256 v0x = P.v0x, v0y = P.v0y;
        const __m256 v1x = P.v1x, v1y = P.v1y;
        const __m256 v2x = P.v2x, v2y = P.v2y;

        const __m256 v0z = P.v0z;
        const __m256 v1z = P.v1z;
        const __m256 v2z = P.v2z;

        const __m256 lightX = P.lightX;
        const __m256 lightY = P.lightY;
        const __m256 lightZ = P.lightZ;

        const __m256 n0x = P.n0x, n0y = P.n0y, n0z = P.n0z;
        const __m256 n1x = P.n1x, n1y = P.n1y, n1z = P.n1z;
        const __m256 n2x = P.n2x, n2y = P.n2y, n2z = P.n2z;

        const __m256 kdVec = P.kdVec;
        const __m256 Lr = P.lightIntensityR, Lg = P.lightIntensityG, Lb = P.lightIntensityB;
        const __m256 Ar = P.ambientLightR, Ag = P.ambientLightG, Ab = P.ambientLightB;

        const __m256 c0r = P.c0r, c0g = P.c0g, c0b = P.c0b;
        const __m256 c1r = P.c1r, c1g = P.c1g, c1b = P.c1b;
        const __m256 c2r = P.c2r, c2g = P.c2g, c2b = P.c2b;

        const __m256 vec255 = P.vec255;
        const __m256 xOffset = P.xOffset;

        for (int y = minY; y < maxY; ++y)
        {

            uint8_t* row = renderer.canvas.rowPtr(y);
            float* zrow = &renderer.zbuffer(0, y);

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
                        float dot = max(vec4::dot(P.omegaN, normal), 0.0f);
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();
                        colour a = (c * kd) * (L.L * dot + (L.ambient * kd));
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        uint8_t* px = row + x * 3;
                        px[0] = r; px[1] = g; px[2] = b;
                        zrow[x] = depth;
                    }
                }
            }

            //process 8 pixels at once
            for (; x <= maxX - 8; x += 8)
            {
                //create x coordinates for 8 pixels
                const __m256 px = _mm256_add_ps(_mm256_set1_ps((float)x), xOffset);

                //calculate barycentric coordinates for 8 pixels
                const __m256 alpha = _mm256_mul_ps(getCoordinatesSIMD(v0x, v0y, v1x, v1y, px, py), invAreaVec);
                const __m256 beta = _mm256_mul_ps(getCoordinatesSIMD(v1x, v1y, v2x, v2y, px, py), invAreaVec);
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
                float* const zrow = &renderer.zbuffer(0, y);         
                const __m256 zbufferDepths = _mm256_loadu_ps(zrow + x);


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

                const __m256 diffuseTermR = _mm256_mul_ps(Lr, dotProduct);
                const __m256 diffuseTermG = _mm256_mul_ps(Lg, dotProduct);
                const __m256 diffuseTermB = _mm256_mul_ps(Lb, dotProduct);

                const __m256 ambientTermR = _mm256_mul_ps(Ar, kdVec);
                const __m256 ambientTermG = _mm256_mul_ps(Ag, kdVec);
                const __m256 ambientTermB = _mm256_mul_ps(Ab, kdVec);

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

                _mm256_maskstore_ps(zrow + x, writeMask, depth);


                if (renderMask != 0)
                {
                    //store the calculated colours into temporary arrays
                    float rArray[8], gArray[8], bArray[8];
                    _mm256_storeu_ps(rArray, cr);
                    _mm256_storeu_ps(gArray, cg);
                    _mm256_storeu_ps(bArray, cb);

                    //loop through the 8 pixels and draw only the ones that passed
                    uint8_t* dst = row + (x * 3);
                    for (int i = 0; i < 8; ++i)
                    {
                        if (renderMask & (1 << i))
                        {
                            const int o = i * 3;
                            dst[o + 0] = (uint8_t)rArray[i];
                            dst[o + 1] = (uint8_t)gArray[i];
                            dst[o + 2] = (uint8_t)bArray[i];
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
                        float dot = max(vec4::dot(P.omegaN, normal), 0.0f);
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();
                        colour a = (c * kd) * (L.L * dot + (L.ambient * kd));
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        uint8_t* px = row + x * 3;
                        px[0] = r; px[1] = g; px[2] = b;
                        zrow[x] = depth;
                    }
                }
            }
        }
    }

    void drawSIMD(Renderer& renderer, Light& L, float ka, float kd)
    {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);

        if (invArea == 0.f) 
            return;

        int minY = (int)minV.y;
        int maxY = (int)ceil(maxV.y);

        if (minY >= maxY) 
            return;


        TriSIMDPre pre;
        buildPre(L, kd, pre);

        pre.minX = (int)minV.x;
        pre.maxX = (int)std::ceil(maxV.x);
        if (pre.minX >= pre.maxX)
            return;

        const int H = maxY - minY;
        const int W = pre.maxX - pre.minX;
        const int pixels = H * W;

        if (invArea == 0.f || H <= 0 || W <= 0 || pixels < 120'000)
        {
            this->drawSIMDrows(renderer, pre, L, ka, kd, minY, maxY);
            return;

        }

        const int rowsPerTask = (W >= 1024 ? 8 : 12);
        const unsigned useThreads = 0;

        tri_internal::RowPool::instance().parallel_rows(minY,maxY,rowsPerTask,[&, this](int y0, int y1) 
        {
            this->drawSIMDrows(renderer, pre, L, ka, kd, y0, y1);
        },
            useThreads
        );
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

