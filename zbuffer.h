#pragma once

#include <concepts>

// Zbuffer class for managing depth values during rendering.
// This class is template-constrained to only work with floating-point types (`float` or `double`).

template<std::floating_point T> // Restricts T to be a floating-point type
class Zbuffer {
    T* buffer;                  // Pointer to the buffer storing depth values
    unsigned int width, height; // Dimensions of the Z-buffer

public:
    // Constructor to initialize a Z-buffer with the given width and height.
    // Allocates memory for the buffer.
    // Input Variables:
    // - w: Width of the Z-buffer.
    // - h: Height of the Z-buffer.
    Zbuffer(unsigned int w, unsigned int h) {
        create(w, h);
    }

    // Default constructor for creating an uninitialized Z-buffer.
    Zbuffer() {
    }

    // Creates or reinitializes the Z-buffer with the given width and height.
    // Allocates memory for the buffer.
    // Input Variables:
    // - w: Width of the Z-buffer.
    // - h: Height of the Z-buffer.
    void create(unsigned int w, unsigned int h) {
        width = w;
        height = h;
        buffer = new T[width * height]; // Allocate memory for the buffer
    }

    // Accesses the depth value at the specified (x, y) coordinate.
    // Input Variables:
    // - x: X-coordinate of the pixel.
    // - y: Y-coordinate of the pixel.
    // Returns a reference to the depth value at (x, y).
    T& operator () (unsigned int x, unsigned int y) {
        return buffer[(y * width) + x]; // Convert 2D coordinates to 1D index
    }

    // Clears the Z-buffer by setting all depth values to 1.0f,
    // which represents the farthest possible depth.
    void clear() {
        for (unsigned int i = 0; i < width * height; i++) {
            buffer[i] = 1.0f; // Reset each depth value
        }
    }

    // Destructor to clean up memory allocated for the Z-buffer.
    ~Zbuffer() {
        delete[] buffer; // Free the allocated memory
    }
};
