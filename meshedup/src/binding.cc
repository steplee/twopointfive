#include <Eigen/StdVector>
#include <torch/extension.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "mc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<Octree>(m, "Octree")
    .def(py::init<const Vector3&, const Vector4i&, scalar, int, int>())
    .def_readwrite("offset", &Octree::offset)
    .def("child", &Octree::child)
    .def("render", &Octree::render)
    .def("info", &Octree::info)
    .def("add", &Octree::add)
    .def("print", &Octree::print)
    //.def("searchNode", &Octree::searchNode)
    .def("search", &Octree::search);

  py::class_<DistIndexPairs>(m, "DistIndexPairs")
    .def_readonly("dists", &DistIndexPairs::dists)
    .def_readonly("indices", &DistIndexPairs::indices)
    .def("__iter__", [](const DistIndexPairs& dip) {
        return py::iter(py::make_tuple(dip.dists, dip.indices));
        })
    ;

  py::class_<IndexedMesh>(m, "IndexedMesh")
    .def_readwrite("verts", &IndexedMesh::verts)
    .def_readwrite("inds", &IndexedMesh::inds)
    .def("print", &IndexedMesh::print)
    .def("render", &IndexedMesh::render);

  m.def("meshOctree", &meshOctree);
  m.def("meshOctreeSurfaceNet", &meshOctreeSurfaceNet);
}
