#pragma once

#include "common.h"

struct Octree {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Octree(const Vector4i& loc, int maxDepth);
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
  int maxDepth;
  Vector3 offset_;

  void add(const Vector3& pt, int id, scalar val);
  void addMany(int id, RowMatrixCRef pts, Eigen::Ref<const Eigen::VectorXf> val);

  Gridcell gridcell() const;

  Octree child(int i);

  std::string info();

  void render(int toDepth);
  void render_(int toDepth);
  void print(int depth);

  //DistIndexPairs search(RowMatrixRef allPts, const RowMatrixRef qpts, int k);
  DistIndexPairs search(const RowMatrixCRef qpts, int k) const;
  Octree* searchNode(const Vector3& pt);
  Octree* searchNode(const Vector4i& loc);

  // False if copied for a python binding (e.g. from searchNode)
  bool mustFree = true;

  Vector3 offset() const;
  scalar size() const;
  inline int depth() const { return loc(3); }

};
