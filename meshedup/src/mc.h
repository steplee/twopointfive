#include <cmath>
#include <Eigen/StdVector>
#include <Eigen/Core>


using scalar = float;
using Vector3 = Eigen::Matrix<scalar,3,1>;
using Vector4i = Eigen::Matrix<int32_t,4,1>;
using Vector4 = Eigen::Matrix<scalar,4,1>;
using RowMatrix = Eigen::Matrix<scalar,-1,-1, Eigen::RowMajor>;
using RowMatrixRef = Eigen::Ref<Eigen::Matrix<scalar,-1,-1, Eigen::RowMajor>>;

struct DistIndexPairs {
  Eigen::Matrix<float, -1,-1> dists;
  Eigen::Matrix<int, -1,-1> indices;
};

using Triangle = Eigen::Matrix<scalar, 3,3, Eigen::RowMajor>;

struct Gridcell {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vector3 p[8];
  scalar val[8];
};

Vector3 VertexInterp(scalar isolevel, Vector3 p1,Vector3 p2, scalar valp1, scalar valp2);
int Polygonise(Gridcell grid, scalar isolevel, Triangle *triangles);


struct Octree {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Octree(const Vector3& offset, const Vector4i& loc, scalar size, int depth, int maxDepth);
  Octree(const Octree& o);
  ~Octree();

  //Vector4 cells[8];
  //int cellIds[8];
  //float cellVals[8];
  int cellId;
  float cellVal;
  Vector3 cellPt;
  Octree* children[8] = {nullptr};

  Vector4i loc;
  Vector3 offset;
  scalar size;
  int depth, maxDepth;

  void add(const Vector3& pt, int id, scalar val);

  Gridcell gridcell();

  Octree child(int i);

  std::string info();

  void render(int toDepth);
  void render_(int toDepth);
  void print(int depth);

  DistIndexPairs search(RowMatrixRef allPts, const RowMatrixRef qpts, int k);
  Octree* searchNode(const Vector3& pt);
  Octree* searchNode(const Vector4i& loc);

  // False if copied for a python binding (e.g. from searchNode)
  bool mustFree = true;

};

struct IndexedMesh {
  std::vector<Eigen::Vector3f> verts;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3f> uvs;
  std::vector<uint32_t> inds;

  uint32_t tex = 0;

  void render();
  void print();
};

std::vector<Triangle> meshOctree(Octree& oct, scalar isolevel);
IndexedMesh meshOctreeSurfaceNet(Octree& oct, scalar isolevel);
