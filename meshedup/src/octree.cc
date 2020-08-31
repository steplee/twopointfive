#include "octree.h"

#include <stack>
#include <iostream>


Octree::Octree(const Vector4i& loc, int maxDepth)
  : maxDepth(maxDepth), loc(loc)
{
  cellId = -1;
  cellVal = 0;
  for (int i=0; i<8; i++)
    children[i] = nullptr;

  offset_.setZero();
  for (int i=0; i<loc(3); i++) {
    int j = i;
    j = loc(3) - i - 1;
    int bit = 1 << j;
    scalar sz2 = scalar(1.) / (1<<(i+1));
    if (loc(0) & bit) offset_(0) += sz2;
    if (loc(1) & bit) offset_(1) += sz2;
    if (loc(2) & bit) offset_(2) += sz2;
  }
}

Octree::Octree(const Octree& o) {
  loc = o.loc;
  maxDepth = o.maxDepth;
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

void Octree::addMany(int id, RowMatrixCRef pts, Eigen::Ref<const Eigen::VectorXf> vals) {
  for (int i=0; i<pts.rows(); i++) {
    add(pts.row(i), id+i, vals(i));
  }
}

void Octree::add(const Vector3& pt, int id, scalar val) {
  // If cellId is -2, that means we cannot the node has children and we cannot just store a record in the node.
  // If cellId is -1, the node has never been touched and is allowed to store the record in itself.
  // If we add() to a node with >=0 cellId, we must create children, add new record, set cellId=-2, then add old stored record.
  //if (cellId == -1) {
  if (depth() == maxDepth) {
    cellPt = pt; cellId = id; cellVal = val;
  } else {
    uint8_t l = 0;
    auto off = offset();
    auto sz = size();
    if (pt(0) > off(0)+sz/2.) l |= 1;
    if (pt(1) > off(1)+sz/2.) l |= 2;
    if (pt(2) > off(2)+sz/2.) l |= 4;
    //std::cout << " add " << pt.transpose() << " check " << off.transpose().array()+(sz/2.) << " -> " << ((int)l) << "\n";

    if (pt(0) > off(0)+sz) return;
    if (pt(1) > off(1)+sz) return;
    if (pt(2) > off(2)+sz) return;

    if (depth() == maxDepth) {
      cellPt = pt;
      cellId = id;
      cellVal = val;
      return;
    }

    if (children[l] == nullptr) {
      Vector4i newloc { loc(0)*2 + (l&1), loc(1)*2 + ((l>>1)&1), loc(2)*2 + ((l>>2)&1), loc(3)+1 };
      children[l] = new Octree(newloc, maxDepth);
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

Vector3 Octree::offset() const {
  return offset_;
  Vector3 off = Vector3::Zero();
  for (int i=0; i<loc(3); i++) {
    int j = i;
    j = loc(3) - i - 1;
    int bit = 1 << j;
    scalar sz2 = scalar(1.) / (1<<(i+1));
    if (loc(0) & bit) off(0) += sz2;
    if (loc(1) & bit) off(1) += sz2;
    if (loc(2) & bit) off(2) += sz2;
    //std::cout << " ::offset (" << loc.transpose() << ") " << i << " " << sz2 << " -> " << off.transpose() << "\n";
  }
  return off;
}
scalar Octree::size() const {
  return scalar(1.) / (1<<loc(3));
}

Gridcell Octree::gridcell() const {
  Gridcell out;

  scalar o = static_cast<scalar>(size()/1.0);
  auto off = offset();
  out.p[0] = off + Vector3 { 0 , 0 , 0 };
  out.p[1] = off + Vector3 { o , 0 , 0 };
  out.p[2] = off + Vector3 { o , o , 0 };
  out.p[3] = off + Vector3 { 0 , o , 0 };

  out.p[4] = off + Vector3 { 0 , 0 , o };
  out.p[5] = off + Vector3 { o , 0 , o };
  out.p[6] = off + Vector3 { o , o , o };
  out.p[7] = off + Vector3 { 0 , o , o };
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
    + std::to_string(offset()(0)) + " "
    + std::to_string(offset()(1)) + " "
    + std::to_string(offset()(2))
    + ", depth=" + std::to_string(depth()) + ")"
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


DistIndexPairs Octree::search(const RowMatrixCRef qpts, int k) const {
  DistIndexPairs out;
  const int n = qpts.rows();
  out.dists.resize(n,k);
  out.indices.resize(n,k);

  #pragma omp parallel for schedule(static)
  for (int ii=0; ii<n; ii++) {
    std::stack<const Octree*> path;
    const Octree* cur = this;
    Vector3 pt = qpts.row(ii);

    while (cur and cur->depth() != cur->maxDepth) {
      path.push(cur);
      int l = 0;
      auto sz = cur->size();
      auto off = cur->offset();
      if (pt(0) > off(0)+sz/2.) l |= 1;
      if (pt(1) > off(1)+sz/2.) l |= 2;
      if (pt(2) > off(2)+sz/2.) l |= 4;
      cur = cur->children[l];
    }

    struct DistIndexScalar { float d; int i; };
    std::vector<DistIndexScalar> pool;

    // One stop criterion is if we aggregate at least K neighbors, then
    // we should not look more than X more levels up.
    // is X 1 or 2?
    //int8_t timesSeenAtleastK = 0, extraLevels = 1;
    int8_t timesSeenAtleastK = 0, extraLevels = 0;

    const Octree* last = nullptr;
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
            if (node->depth() == node->maxDepth) {
              if (node->cellId != -1) {
                //Vector3 cell_pt = allPts.row(node->cellId);
                Vector3 cell_pt = cellPt;
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


void Octree::render(int toDepth) {
  //glBegin(GL_LINES);
  glEnableClientState(GL_VERTEX_ARRAY);
  auto gc = gridcell();
  glVertexPointer(3, GL_FLOAT, 0, &gc.p[0]);
  glMatrixMode(GL_MODELVIEW);
  render_(toDepth);
  glDisableClientState(GL_VERTEX_ARRAY);
  //glEnd();
}

void Octree::render_(int toDepth) {
  if (toDepth == 0) return;

  float alpha = ((float)depth()+1)/((float)maxDepth+4);
  //alpha = depth < maxDepth ? 0 : alpha;
  glColor4f(0,0,1,alpha);

  glPushMatrix();
  float s = size();
  auto off = offset();
  float m[16] = {s,0,0,0, 0,s,0,0, 0,0,s,0, off(0),off(1),off(2),1};
  glMultMatrixf(m);

  //auto gc = gridcell();
  static const int inds[] = { 0,1, 1,2, 2,3, 3,0, 0,4, 1,5, 2,6, 3,7,  4,5, 5,6, 6,7, 7,4};
  //glVertexPointer(3, GL_FLOAT, 0, &gc.p[0]);
  glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, (void*)inds);

  glPopMatrix();

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

  while (cur and cur->depth() != cur->maxDepth) {
    int l = 0;
    float sz = cur->size();
    Vector3 off = cur->offset();
    if (pt(0) > off(0)+sz/2.) l |= 1;
    if (pt(1) > off(1)+sz/2.) l |= 2;
    if (pt(2) > off(2)+sz/2.) l |= 4;
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
