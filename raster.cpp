#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>
#include <vector>

#include <cmath>
#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"

#if defined(__AVX2__)
#pragma message("AVX2 is enabled (__AVX2__ defined)")
#else
#pragma message("AVX2 is NOT enabled")
#endif

struct TriInstance {
    triangle tri;
    float kd;   // per-triangle material (diffuse)
};

// Transform a mesh & append its triangles (screen-space) to 'out'
static void gatherFromMesh(Renderer& renderer,
    Mesh* mesh,
    const matrix& camera,
    std::vector<TriInstance>& out)
{
    matrix p = renderer.perspective * camera * mesh->world;
    out.reserve(out.size() + mesh->triangles.size());

    for (const triIndices& ind : mesh->triangles) {
        Vertex t[3];
        for (unsigned i = 0; i < 3; ++i) {
            t[i].p = p * mesh->vertices[ind.v[i]].p;
            t[i].p.divideW();

            // normals to world space
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // NDC -> screen
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * (float)renderer.canvas.getWidth();
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * (float)renderer.canvas.getHeight();
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1];

            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // very cheap Z-clip
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f)
            continue;

        out.push_back(TriInstance{ triangle(t[0], t[1], t[2]), mesh->kd });
    }
}

// One pool submission for the whole frame.
// Uses row-bands; each task draws ONLY its band across all triangles.
static void drawTrianglesSIMD_Parallel(
    Renderer& renderer,
    std::vector<TriInstance>& tris,
    Light& L)
{
    using tri_internal::RowPool;

    if (tris.empty()) return;

    const int H = renderer.canvas.getHeight();
    const int bandRows = (H >= 720 ? 32 : 16);
    const int numBands = (H + bandRows - 1) / bandRows;

    struct TriJob {
        triangle* t;
        TriSIMDPre pre;
        int y0, y1;
        int minX, maxX;
        float kd;
    };

    // Precompute per-triangle data once
    std::vector<TriJob> jobs;
    jobs.reserve(tris.size());
    for (auto& inst : tris) {
        auto& tri = inst.tri;
        vec2D minV, maxV;
        tri.getBoundsWindow(renderer.canvas, minV, maxV);
        const int y0 = (int)minV.y, y1 = (int)std::ceil(maxV.y);
        const int minX = (int)minV.x, maxX = (int)std::ceil(maxV.x);
        if (y0 >= y1 || minX >= maxX) continue;

        TriJob j;
        j.t = &tri;
        tri.buildPre(L, inst.kd, j.pre);  // kd baked into pre (SIMD)
        j.pre.minX = minX;
        j.pre.maxX = maxX;
        j.y0 = y0; j.y1 = y1;
        j.minX = minX; j.maxX = maxX;
        j.kd = inst.kd;                   // scalar path needs kd too
        jobs.push_back(j);
    }
    if (jobs.empty()) return;

    // Bin triangles into row bands to avoid scanning all tris per task
    std::vector<std::vector<int>> band2tris(numBands);
    for (int i = 0; i < (int)jobs.size(); ++i) {
        const int b0 = max(0, jobs[i].y0 / bandRows);
        const int b1 = min(numBands - 1, (jobs[i].y1 - 1) / bandRows);
        for (int b = b0; b <= b1; ++b) band2tris[b].push_back(i);
    }

    RowPool::instance().parallel_rows(
        /*globalY0=*/0,
        /*globalY1=*/H,
        /*rowsPerTask=*/bandRows,
        /*task*/ [&](int y0, int y1)
        {
            const int band = y0 / bandRows;
            const auto& list = band2tris[band];
            for (int idx : list) {
                const TriJob& j = jobs[idx];
                const int ys = max(y0, j.y0);
                const int ye = min(y1, j.y1);
                if (ys >= ye) continue;
                // NOTE: ka is unused in SIMD path; pass 0.f
                j.t->drawSIMDrows(renderer, j.pre, L, /*ka*/0.f, /*kd*/j.kd, ys, ye);
            }
        },
        /*useThreads=*/0 // 0 = pool default cap (your 11)
    );
}

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) 
{
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;

    std::vector<TriInstance> triangles;
    triangles.reserve(mesh->triangles.size());

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) 
    {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) 
        {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal; 
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        triangles.push_back(TriInstance{ triangle(t[0], t[1], t[2]), mesh->kd });
    }
    if (!triangles.empty())
        drawTrianglesSIMD_Parallel(renderer, triangles, L);
}


// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    // camera is just a matrix
    matrix camera = matrix(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
   // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        // Render each object in the scene
        for (auto& m : scene)
            render(renderer, m, camera, L);

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    bool running = true;

    std::vector<Mesh*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        std::vector<TriInstance> all_triangles_for_frame;
        for (auto& m : scene) {
            gatherFromMesh(renderer, m, camera, all_triangles_for_frame);
        }

        if (!all_triangles_for_frame.empty()) {
            drawTrianglesSIMD_Parallel(renderer, all_triangles_for_frame, L);
        }
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Renderer renderer;
    matrix camera = matrix();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) 
        {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) 
            {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        std::vector<TriInstance> all_triangles_for_frame;
        for (auto& m : scene) {
            gatherFromMesh(renderer, m, camera, all_triangles_for_frame);
        }

        if (!all_triangles_for_frame.empty()) {
            drawTrianglesSIMD_Parallel(renderer, all_triangles_for_frame, L);
        }
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Entry point of the application
// No input variables
int main() 
{
    //Uncomment the desired scene function to run
    scene1();
    //scene2();
    sceneTest(); 
    

    return 0;
}

