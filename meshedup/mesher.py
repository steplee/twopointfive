import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import cv2


def compute_normals(pts):
    size = 10
    t = torch.empty(3).fill_(-size/2)

    #tree = Octree(t,size)
    #tree.place(pts)

    grid = Grid(t,size)
    grid.place(pts)

    pass

# Needs more thought, not really a good fit for this model
class Octree:
    def __init__(self, offset, size, depth=0, res=8):
        self.offset, self.size, self.depth, self.res = offset, size, depth, res
        deep = 4
        self.occ = torch.zeros((res,res,res,deep))
        self.cnt = torch.zeros((res,res,res),dtype=np.int32)

    def place(self, pts):
        pts = (pts - self.offset) / self.size
        assn = pts // self.res


    def search(self, pts):
        pass

class Grid:
    def __init__(self, offset, size, res=32):
        self.offset, self.size, self.res = offset, size, res
        self.grid_ = torch.zeros((res,res,res,3))
        self.grid = torch.zeros((res,res,res,3))
        self.cnts = torch.zeros((res,res,res,1))

    def place(self, pts0):
        pts = (pts0 - self.offset) / self.size
        coo = (pts // self.res).T.to(torch.int64)

        vals = pts0
        sparseSize3 = torch.Size((self.res,self.res,self.res,3))
        sparseSize1 = torch.Size((self.res,self.res,self.res,1))

        self.grid_ += torch.sparse.FloatTensor(coo, vals, sparseSize3).to_dense()
        self.cnts += torch.sparse.FloatTensor(coo, torch.ones_like(vals[...,:1]), sparseSize1).to_dense()
        self.grid = self.grid_ / self.cnts.clamp(1)

    def search(self, pts, k=2):
        self.pts


class Mesher():
    def __init__(self, meta):
        for k,v in meta.items(): setattr(self,k,v)

    def run(self, pts,nls):
        pass
