#pragma once

#include <cmath>
#include <Eigen/StdVector>
#include <Eigen/Core>

#include <GL/glew.h>

// Assert but even when debugging
#define ENSURE(x)                                                                                  \
    do {                                                                                                 \
        if (!(x)) {                                                                                      \
            std::cout << " - enusre failed " << __LINE__ << " " << __FILE__ << " : " << #x << std::endl; \
            exit(1);                                                                                     \
        }                                                                                                \
    } while (0);

using scalar = float;
using Vector3 = Eigen::Matrix<scalar,3,1>;
using Vector4i = Eigen::Matrix<int32_t,4,1>;
using Vector4 = Eigen::Matrix<scalar,4,1>;
using RowMatrix = Eigen::Matrix<scalar,-1,-1, Eigen::RowMajor>;
using RowMatrixRef = Eigen::Ref<Eigen::Matrix<scalar,-1,-1, Eigen::RowMajor>>;
using RowMatrixCRef = Eigen::Ref<const Eigen::Matrix<scalar,-1,-1, Eigen::RowMajor>>;


struct Gridcell {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vector3 p[8];
  scalar val[8];
};

struct DistIndexPairs {
  Eigen::Matrix<float, -1,-1> dists;
  Eigen::Matrix<int, -1,-1> indices;
};
