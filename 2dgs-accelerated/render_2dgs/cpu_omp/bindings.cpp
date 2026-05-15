#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <omp.h>

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::array_t<float> render_cpu_omp(FloatArray, FloatArray, FloatArray,
                                  FloatArray, FloatArray, int, int);

PYBIND11_MODULE(render_2dgs_cpu_omp, m) {
    m.doc() = "2DGS CPU renderer — OpenMP tile-parallel (TILE_H=32, TILE_W=32)";

    m.def("render", &render_cpu_omp,
          py::arg("mus"), py::arg("sigmas"), py::arg("thetas"),
          py::arg("opacities"), py::arg("rgbs"),
          py::arg("H"), py::arg("W"),
          "Render 2D Gaussians on CPU with OpenMP.\n"
          "Inputs : float32 ndarray (N,2), (N,2), (N,), (N,), (N,3)\n"
          "Returns: float32 ndarray (H, W, 3)");

    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("n"),
          "Set the number of OpenMP threads for subsequent render() calls.");
}
