#include "mc.h"
#include <stack>
#include <iostream>
#include <unordered_set>


// http://paulbourke.net/geometry/polygonise/

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
   */
Vector3 VertexInterp(scalar isolevel, Vector3 p1,Vector3 p2, scalar valp1, scalar valp2)
{
  scalar mu;
  Vector3 p;

  if (abs(isolevel-valp1) < 0.00001)
    return(p1);
  if (abs(isolevel-valp2) < 0.00001)
    return(p2);
  if (abs(valp1-valp2) < 0.00001)
    return(p1);

  mu = (isolevel - valp1) / (valp2 - valp1);
  p(0) = p1(0) + mu * (p2(0) - p1(0));
  p(1) = p1(1) + mu * (p2(1) - p1(1));
  p(2) = p1(2) + mu * (p2(2) - p1(2));

  return p;
}

/*
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
   0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
   */
int Polygonise(Gridcell grid, scalar isolevel, Triangle *triangles)
{
  int i,ntriang;
  int cubeindex;
  Vector3 vertlist[12];

  int edgeTable[256]={
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
  int triTable[256][16] =
  {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

  /*
     Determine the index into the edge table which
     tells us which vertices are inside of the surface
     */
  cubeindex = 0;
  if (grid.val[0] < isolevel) cubeindex |= 1;
  if (grid.val[1] < isolevel) cubeindex |= 2;
  if (grid.val[2] < isolevel) cubeindex |= 4;
  if (grid.val[3] < isolevel) cubeindex |= 8;
  if (grid.val[4] < isolevel) cubeindex |= 16;
  if (grid.val[5] < isolevel) cubeindex |= 32;
  if (grid.val[6] < isolevel) cubeindex |= 64;
  if (grid.val[7] < isolevel) cubeindex |= 128;

  /* Cube is entirely in/out of the surface */
  if (edgeTable[cubeindex] == 0)
    return(0);

  /* Find the vertices where the surface intersects the cube */
  if (edgeTable[cubeindex] & 1)
    vertlist[0] =
      VertexInterp(isolevel,grid.p[0],grid.p[1],grid.val[0],grid.val[1]);
  if (edgeTable[cubeindex] & 2)
    vertlist[1] =
      VertexInterp(isolevel,grid.p[1],grid.p[2],grid.val[1],grid.val[2]);
  if (edgeTable[cubeindex] & 4)
    vertlist[2] =
      VertexInterp(isolevel,grid.p[2],grid.p[3],grid.val[2],grid.val[3]);
  if (edgeTable[cubeindex] & 8)
    vertlist[3] =
      VertexInterp(isolevel,grid.p[3],grid.p[0],grid.val[3],grid.val[0]);
  if (edgeTable[cubeindex] & 16)
    vertlist[4] =
      VertexInterp(isolevel,grid.p[4],grid.p[5],grid.val[4],grid.val[5]);
  if (edgeTable[cubeindex] & 32)
    vertlist[5] =
      VertexInterp(isolevel,grid.p[5],grid.p[6],grid.val[5],grid.val[6]);
  if (edgeTable[cubeindex] & 64)
    vertlist[6] =
      VertexInterp(isolevel,grid.p[6],grid.p[7],grid.val[6],grid.val[7]);
  if (edgeTable[cubeindex] & 128)
    vertlist[7] =
      VertexInterp(isolevel,grid.p[7],grid.p[4],grid.val[7],grid.val[4]);
  if (edgeTable[cubeindex] & 256)
    vertlist[8] =
      VertexInterp(isolevel,grid.p[0],grid.p[4],grid.val[0],grid.val[4]);
  if (edgeTable[cubeindex] & 512)
    vertlist[9] =
      VertexInterp(isolevel,grid.p[1],grid.p[5],grid.val[1],grid.val[5]);
  if (edgeTable[cubeindex] & 1024)
    vertlist[10] =
      VertexInterp(isolevel,grid.p[2],grid.p[6],grid.val[2],grid.val[6]);
  if (edgeTable[cubeindex] & 2048)
    vertlist[11] =
      VertexInterp(isolevel,grid.p[3],grid.p[7],grid.val[3],grid.val[7]);

  /* Create the triangle */
  ntriang = 0;
  for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
    triangles[ntriang].row(0) = vertlist[triTable[cubeindex][i  ]];
    triangles[ntriang].row(1) = vertlist[triTable[cubeindex][i+1]];
    triangles[ntriang].row(2) = vertlist[triTable[cubeindex][i+2]];
    ntriang++;
  }

  return ntriang;
}



Octree::Octree(const Vector3& offset, const Vector4i& loc, scalar size, int depth, int maxDepth)
  : offset(offset), size(size), depth(depth), maxDepth(maxDepth), loc(loc)
{
  cellId = -1;
  cellVal = 0;
  for (int i=0; i<8; i++)
    children[i] = nullptr;
}

Octree::Octree(const Octree& o) {
  for (int i=0; i<8; i++) {
    cellId = o.cellId;
    cellVal = o.cellVal;
    children[i] = o.children[i];
  }
}

Octree::~Octree() {
  //std::cout << " - destructor -- must free: " << mustFree << "\n";
  if (mustFree) {
    for (int i=0; i<8; i++) {
      if (children[i]) {
        delete children[i];
        children[i] = nullptr;
      }
  }
}
}

void Octree::add(const Vector3& pt, int id, scalar val) {
  // If cellId is -2, that means we cannot the node has children and we cannot just store a record in the node.
  // If cellId is -1, the node has never been touched and is allowed to store the record in itself.
  // If we add() to a node with >=0 cellId, we must create children, add new record, set cellId=-2, then add old stored record.
  //if (cellId == -1) {
  if (depth == maxDepth) {
    cellPt = pt; cellId = id; cellVal = val;
  } else {
    uint8_t l = 0;
    if (pt(0) > offset(0)+size/2.) l |= 1;
    if (pt(1) > offset(1)+size/2.) l |= 2;
    if (pt(2) > offset(2)+size/2.) l |= 4;

    if (pt(0) > offset(0)+size) return;
    if (pt(1) > offset(1)+size) return;
    if (pt(2) > offset(2)+size) return;

    if (depth == maxDepth) {
      cellPt = pt;
      cellId = id;
      cellVal = val;
      return;
    }

    if (children[l] == nullptr) {
      Vector3 new_off = Vector3::Zero();
      Vector4i newloc { loc(0)*2, loc(1)*2, loc(2)*2, loc(3)+1 };
      if (l & 1) { new_off(0) += size/2.0; newloc(0) += 1; }
      if (l & 2) { new_off(1) += size/2.0; newloc(1) += 1; }
      if (l & 4) { new_off(2) += size/2.0; newloc(2) += 1; }
      children[l] = new Octree(offset + new_off, newloc, size/2.0, depth+1, maxDepth);
    }
    children[l]->add(pt, id, val);

    // If cellId != -2, we must split, than insert old point.
    if (cellId != -2) {
      Vector3 pt__ = cellPt;
      int id__ = cellId;
      scalar val__ = cellVal;
      cellPt.setZero(); cellId = -2; cellVal = 0;
      this->add(pt__, id__, val__);
    }
  }

}

Gridcell Octree::gridcell() {
  Gridcell out;

  scalar o = static_cast<scalar>(size/1.0);
  out.p[0] = offset + Vector3 { 0 , 0 , 0 };
  out.p[1] = offset + Vector3 { o , 0 , 0 };
  out.p[2] = offset + Vector3 { o , o , 0 };
  out.p[3] = offset + Vector3 { 0 , o , 0 };

  out.p[4] = offset + Vector3 { 0 , 0 , o };
  out.p[5] = offset + Vector3 { o , 0 , o };
  out.p[6] = offset + Vector3 { o , o , o };
  out.p[7] = offset + Vector3 { 0 , o , o };
  for (int i=0; i<8; i++) out.val[i] = cellVal;


  /*
  scalar o = static_cast<scalar>(size/1.0);
  out.p[0] = offset + Vector3 { 0 , 0 , 0 };
  out.p[1] = offset + Vector3 { o , 0 , 0 };
  out.p[2] = offset + Vector3 { o , o , 0 };
  out.p[3] = offset + Vector3 { 0 , o , 0 };

  out.p[4] = offset + Vector3 { 0 , 0 , o };
  out.p[5] = offset + Vector3 { o , 0 , o };
  out.p[6] = offset + Vector3 { o , o , o };
  out.p[7] = offset + Vector3 { 0 , o , o };

  //for (int i=0;i<8;i++) std::swap(out.p[i](1), out.p[i](2));
  //for (int i=0;i<8;i++) out.p[i](2) *= -1;
  //for (int i=0;i<8;i++) out.p[i](2) *= -1;


  //for (int i=0; i<8; i++) out.val[i] = cells[i](3);
  for (int i=0; i<8; i++) out.val[i] = cellVals[i];
  */

  return out;
}

std::string Octree::info() {
  int nc = 0;
  for (int i=0; i<8; i++) if (children[i]) nc++;
  std::string out = " - Node at ("
    + std::to_string(offset(0)) + " "
    + std::to_string(offset(1)) + " "
    + std::to_string(offset(2))
    + ", depth=" + std::to_string(depth) + ")"
    + " with " + std::to_string(nc) + " children";
  std::cout << "    - cell id/val: " << cellId << " " << cellVal << "\n";
  return out;
}

void Octree::print(int depth) {
  for (int i=0; i<depth; i++) std::cout << "  ";
  std::cout << " node (" << loc.transpose() << ") with id " << cellId << " val " << cellVal << "\n";
  for (int i=0; i<8; i++) {
    if (children[i]) children[i]->print(depth+1);
  }
}


DistIndexPairs Octree::search(const RowMatrixRef allPts, const RowMatrixRef qpts, int k) {
  DistIndexPairs out;
  int n = qpts.rows();
  out.dists.resize(n,k);
  out.indices.resize(n,k);

  for (int ii=0; ii<n; ii++) {
    std::stack<Octree*> path;
    Octree* cur = this;
    Vector3 pt = qpts.row(ii);

    while (cur and cur->depth != cur->maxDepth) {
      path.push(cur);
      int l = 0;
      if (pt(0) > cur->offset(0)+cur->size/2.) l |= 1;
      if (pt(1) > cur->offset(1)+cur->size/2.) l |= 2;
      if (pt(2) > cur->offset(2)+cur->size/2.) l |= 4;
      cur = cur->children[l];
    }

    struct DistIndexScalar { float d; int i; };
    std::vector<DistIndexScalar> pool;

    // One stop criterion is if we aggregate at least K neighbors, then
    // we should not look more than X more levels up.
    // is X 1 or 2?
    int8_t timesSeenAtleastK = 0, extraLevels = 1;

    Octree* last = nullptr;
    while ( (not path.empty()) and (timesSeenAtleastK <= extraLevels)) {
      cur = path.top();
      path.pop();
      bool didAdd = false;

      // DFS from this node, excepting 'last' visited node;
      for (int i=0; i<8; i++) {
        if (cur->children[i] and cur->children[i] != last) {
          std::stack<Octree*> st; st.push(cur->children[i]);
          while (not st.empty()) {
            auto node = st.top(); st.pop();
            if (node->depth == node->maxDepth) {
              if (node->cellId != -1) {
                Vector3 cell_pt = allPts.row(node->cellId);
                pool.push_back( DistIndexScalar{(cell_pt-pt).squaredNorm(), node->cellId} );
                didAdd = true;
              }
            } else for (int j=0; j<8; j++) if (node->children[j]) st.push(node->children[j]);
          }
        }
      }
      last = cur;

      if (didAdd) {
        std::sort(pool.begin(), pool.end(), [&pt](const DistIndexScalar &a, const DistIndexScalar &b) { return a.d < b.d; });
        if (pool.size() > k) {
          pool.resize(k);
        }
      }
      if (pool.size() >= k) timesSeenAtleastK++;
    }

    //std::cout << " - Got Pool of size " << pool.size() << ":\n";
    //for (int i=0; i<pool.size(); i++) { std::cout << "(" << pool[i].d << " " << pool[i].i << ") "; }

    int size = pool.size() < k ? pool.size() : k;
    for (int kk=0; kk<size; kk++) {
      out.dists(ii,kk) = pool[kk].d;
      out.indices(ii,kk) = pool[kk].i;
    }
    for (int kk=size; kk<k; kk++) {
      out.dists(ii,kk) = 9e12;
      out.indices(ii,kk) = -1;
    }
  }

  //if (out.size() > k) out.resize(k);
  return out;
}

//std::vector<Triangle> meshOctree(Octree& oct, scalar isolevel) { }

std::vector<Triangle> meshOctree(Octree& oct, scalar isolevel) {
  std::vector<Triangle> out;

  // Traverse octree, for now data is only stored at maxDepth, so visit them and get tris.
  std::stack<Octree*> st;
  std::vector<Octree*> path;
  st.push(&oct);

  int nleaves_ = 0, ntris_ = 0;
  int nbad=0, ngood=0;

  while (not st.empty()) {
    Octree* node = st.top();
    st.pop();

    if (node->depth == node->maxDepth) {
      Triangle tmpTris[5];

      //if (node->loc(2) % 2 == 1) continue;
      //if (node->loc(0) % 2 == 1) continue;
      //if (node->loc(1) % 2 == 1) continue;

      //Gridcell gc = node->gridcell();
      Gridcell gc;
      //gc.p[0] = node->offset.array() + ((float)node->size/2.);
      //gc.p[0] = node->offset;
      //gc.val[0] = node->cellVal;
      bool bad = false;
      for (int i=0; i<2; i++)
      for (int j=0; j<2; j++)
      for (int k=0; k<2; k++) {
        //if (i == 0 and j == 0 and k == 0) continue;
        //Vector4i the_loc = node->loc - Vector4i{i,j,k,0};
        //if (the_loc(0)<0 or the_loc(1)<0 or the_loc(2)<0) { bad = true; continue; }
        //if (i == 0 and j == 0 and k == 0) continue;
        Vector4i the_loc = node->loc + Vector4i{i,j,k,0};

        int lll = 0;
        if (i==0 and j==0) lll = 0;
        if (i==1 and j==0) lll = 1;
        if (i==1 and j==1) lll = 2;
        if (i==0 and j==1) lll = 3;
        int ll = (k<<2) + lll;

        Octree* sib = oct.searchNode(the_loc);
        if (sib->loc == the_loc) {
          gc.val[ll] = sib->cellVal;
          gc.p[ll] = sib->offset.array() + node->size/2.;
        } else {
          gc.val[ll] = 0;
          gc.p[ll] = node->offset + Vector3 { i*node->size , j*node->size , k*node->size };
        }
      }


      if (not bad) {
        int ntris = Polygonise(gc, isolevel, tmpTris);
        ntris_ += ntris;
        nleaves_ += 1;
        ngood++;
        for (int i=0; i<ntris; i++) {
          out.push_back(tmpTris[i]);
        }
      } else {
        nbad++;
        //std::cout << " bad at " << node->loc.transpose() << "\n";
      }

    } else {
      path.push_back(node);
      for (int i=0; i<8; i++) if (node->children[i]) st.push(node->children[i]);
      path.pop_back();
    }
  }

  std::cout << " - " << ntris_ << " tris from " << nleaves_ << " leaves (nbad/ntotal " << nbad << "/" << (ngood+nbad) << ")\n";

  return out;
}


IndexedMesh meshOctreeSurfaceNet(Octree& oct, scalar iso) {
  IndexedMesh out;

  // Traverse octree, for now data is only stored at maxDepth, so visit them and get tris.
  std::stack<Octree*> st;
  std::vector<Octree*> path;
  st.push(&oct);

  int nleaves_ = 0, ntris_ = 0;
  int nbad=0, ngood=0;


  while (not st.empty()) {
    Octree* node = st.top();
    st.pop();

    if (node->depth == node->maxDepth) {
      Triangle tmpTris[5];
      Gridcell gc;
      bool bad = false;
      for (int i=0; i<2; i++)
      for (int j=0; j<2; j++)
      for (int k=0; k<2; k++) {
        Vector4i the_loc = node->loc + Vector4i{i,j,k,0};

        Octree* sib = oct.searchNode(the_loc);
        if (sib->loc == the_loc) {
          int ll = (k<<2) + (j<<1) + i;
          gc.val[ll] = sib->cellVal;
          gc.p[ll] = sib->offset;
        } else bad = true;
      }

      if (not bad) {

        int edges[12][2] = {
          { 0b001 , 0b000 },
          { 0b010 , 0b000 },
          { 0b011 , 0b001 },
          { 0b011 , 0b010 },
          { 0b100 , 0b000 },
          { 0b101 , 0b001 },
          { 0b101 , 0b100 },
          { 0b110 , 0b010 },
          { 0b110 , 0b100 },
          { 0b111 , 0b011 },
          { 0b111 , 0b101 },
          { 0b111 , 0b110 },
        };

        int e_count = 0;
        Vector3 vert = Vector3::Zero();

        // For every edge of cube
        for (int ii=0; ii<12; ii++) {
          int u = edges[ii][0], v = edges[ii][1];
          float pu = gc.val[u], pv = gc.val[v];
          bool signu = pu>iso, signv = pv > iso;
          if (signu != signv) {
            e_count++;
            float t = pu / (pu-pv);
            for (int j=0; j<3; j++) {
              int a = u & (1<<j), b = v & (1<<j);
              if (a != b) vert(j) += a ? 1. - t : t;
              else        vert(j) += a ? 1. : 0.;
            }
          }
        }

        vert = node->offset + vert * (node->size / e_count);

        out.verts.push_back(vert);
        // TODO record index.

        for (int i=0; i<3; i++) {

        }

      } else {
        nbad++;
      }

    } else {
      path.push_back(node);
      for (int i=0; i<8; i++) if (node->children[i]) st.push(node->children[i]);
      path.pop_back();
    }
  }

  std::cout << " - " << ntris_ << " tris from " << nleaves_ << " leaves (nbad/ntotal " << nbad << "/" << (ngood+nbad) << ")\n";

  return out;
}
















#include <GL/glew.h>
void Octree::render(int toDepth) {
  //glBegin(GL_LINES);
  glEnableClientState(GL_VERTEX_ARRAY);
  render_(toDepth);
  glDisableClientState(GL_VERTEX_ARRAY);
  //glEnd();
}

void Octree::render_(int toDepth) {
  if (toDepth == 0) return;

  float alpha = ((float)depth+1)/((float)maxDepth+4);
  //alpha = depth < maxDepth ? 0 : alpha;
  glColor4f(0,0,1,alpha);

  auto gc = gridcell();
  static const int inds[] = { 0,1, 1,2, 2,3, 3,0, 0,4, 1,5, 2,6, 3,7,  4,5, 5,6, 6,7, 7,4};
  /*
  for (int i=0; i<24; i++)
    glVertex3f(gc.p[inds[i]](0), gc.p[inds[i]](1), gc.p[inds[i]](2));
  */
    glVertexPointer(3, GL_FLOAT, 0, &gc.p[0]);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, (void*)inds);

  for (int j=0; j<8; j++)
    if (this->children[j] != nullptr) {
      this->children[j]->render_(toDepth-1);
    }

}

Octree Octree::child(int i) {
  Octree node (*children[i]);
  node.mustFree = false;
  return node;
}
Octree* Octree::searchNode(const Vector3& pt) {
  Octree* cur = this;

  while (cur and cur->depth != cur->maxDepth) {
    int l = 0;
    if (pt(0) > cur->offset(0)+cur->size/2.) l |= 1;
    if (pt(1) > cur->offset(1)+cur->size/2.) l |= 2;
    if (pt(2) > cur->offset(2)+cur->size/2.) l |= 4;
    cur = cur->children[l];
  }
  return cur;
}

Octree* Octree::searchNode(const Vector4i& loc_) {
  Octree* cur = this;
  int qx=loc_(0), qy=loc_(1), qz=loc_(2), qd=loc_(3);

  //std::cout << " searching for " << loc_.transpose() << "\n";

  while (true) {
    int l = 0;
    int x=cur->loc(0), y=cur->loc(1), z=cur->loc(2), dd=cur->loc(3);

    int d = qd - dd - 1;

    if (qx & (1<<d)) l |= 1;
    if (qy & (1<<d)) l |= 2;
    if (qz & (1<<d)) l |= 4;

    //std::cout << "   at " << cur->loc.transpose() << " going " << ((l)&1) << " " << ((l>>1)&1) << " " << ((l>>2)&1) << " : " << l << "\n";

    if (cur->children[l])
      cur = cur->children[l];
    else {
      //std::cout << "   end at " << cur->loc.transpose() << "\n";
      return cur;
    }
  }
}

void IndexedMesh::print() {
  std::cout << " - Mesh (" << verts.size() << " verts) (" << inds.size() << " inds)" << "\n";
}
void IndexedMesh::render() {
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, &verts[0]);
  if (normals.size()) {
    assert(normals.size() == verts.size());
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, &normals[0]);
  }
  if (uvs.size()) {
    assert(uvs.size() == verts.size());
    glClientActiveTexture(GL_TEXTURE0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glTexCoordPointer(2, GL_FLOAT, 0, &uvs[0]);

  }
  glDrawElements(GL_LINES, inds.size(), GL_UNSIGNED_INT, (void*)inds[0]);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}
