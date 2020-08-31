#pragma once

#include "common.h"

struct Octree;


using Triangle = Eigen::Matrix<scalar, 3,3, Eigen::RowMajor>;


Vector3 VertexInterp(scalar isolevel, Vector3 p1,Vector3 p2, scalar valp1, scalar valp2);
int Polygonise(Gridcell grid, scalar isolevel, Triangle *triangles);


struct IndexedMesh {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RowMatrix verts, vertexNormals, faceNormals, uvs;
  std::vector<uint32_t> inds;

  GLuint tex = 0;
  int vertCntr = 0;
  GLenum mode = GL_TRIANGLES;

  void render();
  void print();
};

IndexedMesh meshOctree(Octree& oct, scalar isolevel);
IndexedMesh meshOctreeSurfaceNet(Octree& oct, scalar isolevel);

// Create a mesh to display the edges of a triangular mesh.
IndexedMesh convertTriangleMeshToLines(const IndexedMesh&);
// Create a mesh to display the normals of a point set.
IndexedMesh normalsToMesh(const RowMatrixCRef verts, const RowMatrixCRef normals, float size);

// Given a triangular mesh, average the face normals to get vertex normals.
void computeVertexNormals(Octree& tree, IndexedMesh& mesh);

// For every qpt, approximate the normal by searching the tree+treePts for top
// few neighbors and taking the least important principal axis.
RowMatrix computePointSetNormals(const Octree& tree, RowMatrixCRef treePts, RowMatrixCRef qpts);


// For every point+normal, return up-to two new points at both ends of the (unoriented) normal.
// Points are NOT created if the closest existing point to them is not the one owning the normal (this
// is to prevent intersections)
// Example in 1D:
//   o    =>  x--o--x
//   but
//   o-o  =>  x--o-o--x (inner negative is not created)
RowMatrix getNegativePoints(
    const Octree& tree,
    RowMatrixCRef treePts,
    RowMatrixCRef treePtNormals,
    float normalLength);
