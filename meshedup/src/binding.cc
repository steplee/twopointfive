#include <Eigen/StdVector>
#include <torch/extension.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "mc.h"
#include "octree.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<Octree>(m, "Octree")
    //.def(py::init<const Vector3&, const Vector4i&, scalar, int, int>())
    //.def_readwrite("offset", &Octree::offset)
    .def(py::init<const Vector4i&, int>())
    .def("offset", &Octree::offset)
    .def("size", &Octree::size)
    .def("child", &Octree::child)
    .def("render", &Octree::render)
    .def("info", &Octree::info)
    .def("add", &Octree::add)
    .def("addMany", &Octree::addMany)
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
    .def_readwrite("vertexNormals", &IndexedMesh::vertexNormals)
    .def_readwrite("faceNormals", &IndexedMesh::faceNormals)
    .def_readwrite("uvs", &IndexedMesh::uvs)
    .def_readwrite("tex", &IndexedMesh::tex)
    .def("print", &IndexedMesh::print)
    .def("render", &IndexedMesh::render)
    /*.def("verts", [](IndexedMesh& self) {
        return py::array_t<float>(
            {(ssize_t)self.verts.size(), (ssize_t)3},
            (float*)self.verts.data());
    });*/
    ;

  m.def("meshOctree", &meshOctree);
  m.def("meshOctreeSurfaceNet", &meshOctreeSurfaceNet);
  m.def("convertTriangleMeshToLines", &convertTriangleMeshToLines);
  m.def("normalsToMesh", &normalsToMesh);
  m.def("computeVertexNormals", &computeVertexNormals);
  m.def("computePointSetNormals", &computePointSetNormals);
  m.def("getNegativePoints", &getNegativePoints);

}
