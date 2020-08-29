import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2, time

from OpenGL.GL import *
from OpenGL.GLUT import *
from .gl_stuff import *
import pycmeshedup

# patchBbox is in native coords (probably UTM)
def get_tiff_patch(tiffName, patchBbox, res):
    import tview
    dset = tview.GeoDataset(tiffName)
    aspect_hw = patchBbox[3] / patchBbox[2]
    print(' - Dset bbox:', patchBbox)
    img = np.array(dset.bboxNative(patchBbox, 2048,int(aspect_hw*2048), True))
    #cv2.imshow('tex',tex);cv2.waitKey(0)
    return img

class Meshing():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.setdefault('iso', 1e-6)
        self.cfg.setdefault('maxDepth', 10)

    def run(self, meta):
        pts,vals = meta['pts'], meta['vals']
        loc = np.zeros(4,dtype=np.int32)
        st = time.time()
        offset = -np.ones(3, dtype=np.float32) / 2.
        size  =1
        tree = pycmeshedup.Octree(offset, loc, size, 0, self.cfg['maxDepth'])
        id = 0
        for (pt,val) in (zip(pts,vals)):
            tree.add(pt, id, val)
            id += 1
        print(' - Octree indexing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

        st = time.time()
        mesh = pycmeshedup.meshOctree(tree, self.cfg['iso'])
        print(' - Octree meshing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

        return tree, mesh

def get_dc_lidar_data():
    import pylas, ctypes
    stride = 1
    #f = '/data/lidar/USGS_LPC_VA_Fairfax_County_2018_e1617n1924.laz'
    f,stride = '/data/lidar/dc1.las', 2
    #f,stride = '/data/lidar/PA_Statewide_S_2006-2008_002771.las', 2
    #f,stride = '/data/lidar/airport.las', 4

    st0 = time.time()
    with pylas.open(f) as fh:
        N = -1
        st1 = time.time()
        las = fh.read()
        print('   - las read     took {:.2f}ms'.format((time.time()-st1)*1000))
        st1 = time.time()
        x,y,z = las.x[0:N:stride],las.y[:N:stride],las.z[:N:stride]
        print('   - las slice    took {:.2f}ms'.format((time.time()-st1)*1000))
        st1 = time.time()
        (x1,x2),(y1,y2),(z1,z2) = np.quantile(x[::4],[.1,.9]), np.quantile(y[::4],[.1,.9]), np.quantile(z[::4],[.1,.9])
        print('   - las quantile took {:.2f}ms'.format((time.time()-st1)*1000))
        modelScale = x2-x1
        xx,yy,zz = x - x1 - modelScale/2, y - y1 - modelScale/2, z - z1

        st1 = time.time()
        pts = np.stack((xx,yy,zz), -1).astype(np.float32)
        print('   - las stack    took {:.2f}ms'.format((time.time()-st1)*1000))
        pts = pts / modelScale
        size = 2
        print(pts[::100000])
    print(' - las total load took {:.2f}ms'.format((time.time()-st0)*1000))

    vals = np.ones_like(pts[:,0])

    M = np.eye(4, dtype=np.float32)

    coords = (x1,x2, y1,y2, z1,z2, modelScale)
    return dict(
            pts=pts,
            vals=vals,
            M=M,
            tiff='/data/dc_tiffs/dc.tif'
            )

if __name__ == '__main__':
    cfg = {}
    meshing = Meshing(cfg)

    meta = get_dc_lidar_data()
    meshing.run(meta)
