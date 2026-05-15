#include <algorithm>
#include <cmath>

#include <omp.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

constexpr int TILE_H = 32;
constexpr int TILE_W = 32;

// render_tile: process a single TILE_H x TILE_W pixel block.
// Pixel-outer / Gaussian-inner loop mirroring render.cu lines 155-175.
static void render_tile(
    float*       img,
    const float* mus,
    const float* sigmas,
    const float* thetas,
    const float* opacities,
    const float* rgbs,
    int N, int H, int W,
    int tile_row, int tile_col
) {
    int py0 = tile_row * TILE_H;
    int py1 = std::min(py0 + TILE_H, H);
    int px0 = tile_col * TILE_W;
    int px1 = std::min(px0 + TILE_W, W);

    for (int py = py0; py < py1; ++py) {
        // Pixel coordinate -> normalized [-1, 1]
        // y-flip is intentional: y = 1 - (py + 0.5) / H * 2
        float y = 1.f - (py + 0.5f) / H * 2.f;

        for (int px = px0; px < px1; ++px) {
            float x = (px + 0.5f) / W * 2.f - 1.f;

            float r = 0.f, g = 0.f, b = 0.f;

            for (int i = 0; i < N; ++i) {
                float ct = std::cos(thetas[i]);
                float st = std::sin(thetas[i]);
                float dx = x - mus[i * 2];
                float dy = y - mus[i * 2 + 1];

                // Rotate into Gaussian's local frame
                float rx =  ct * dx + st * dy;
                float ry = -st * dx + ct * dy;

                float sx = sigmas[i * 2];
                float sy = sigmas[i * 2 + 1];
                float exp_val = std::exp(-0.5f * (rx * rx / (sx * sx)
                                                + ry * ry / (sy * sy)));
                float alpha = opacities[i] * exp_val;

                r += alpha * rgbs[i * 3];
                g += alpha * rgbs[i * 3 + 1];
                b += alpha * rgbs[i * 3 + 2];
            }

            int idx = (py * W + px) * 3;
            img[idx]     = std::fmin(r, 1.f);
            img[idx + 1] = std::fmin(g, 1.f);
            img[idx + 2] = std::fmin(b, 1.f);
        }
    }
}

py::array_t<float> render_cpu_omp(
    FloatArray mus, FloatArray sigmas, FloatArray thetas,
    FloatArray opacities, FloatArray rgbs,
    int H, int W
) {
    int N = static_cast<int>(mus.request().shape[0]);

    const float* p_mus      = mus.data();
    const float* p_sigmas   = sigmas.data();
    const float* p_thetas   = thetas.data();
    const float* p_opacities = opacities.data();
    const float* p_rgbs     = rgbs.data();

    py::array_t<float> out({H, W, 3});
    float* img = out.mutable_data();

    // Zero-init before parallel section
    std::fill(img, img + static_cast<size_t>(H) * W * 3, 0.f);

    int n_tile_rows = (H + TILE_H - 1) / TILE_H;
    int n_tile_cols = (W + TILE_W - 1) / TILE_W;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int tr = 0; tr < n_tile_rows; ++tr) {
        for (int tc = 0; tc < n_tile_cols; ++tc) {
            render_tile(img, p_mus, p_sigmas, p_thetas, p_opacities, p_rgbs,
                        N, H, W, tr, tc);
        }
    }

    return out;
}
