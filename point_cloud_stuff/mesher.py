import gdal
import random
import numpy as np
import argparse
import os, sys, time
import time
import pylas
from skimage.morphology import skeletonize
from skimage import measure

import cv2
from matplotlib.cm import inferno


def normalize_and_show(arr):
    dimg = (inferno((arr - arr.min()) / (arr.max()-arr.min())) * 255)[...,0:3].astype(np.uint8)
    dimg = cv2.cvtColor(dimg,cv2.COLOR_RGB2BGR)
    cv2.imshow('dimg', dimg); cv2.waitKey(0); cv2.destroyAllWindows()

# Find a triangular mesh for a DTED tiff.
# The tiff has a cellular structure which yields a mesh, but
# we want to reduce the number of tris.
def mesh_tiff(img):
    #img = cv2.GaussianBlur(img, (5,5), 2)

    dx = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    l = np.sqrt(dx**2+dy**2)

    '''himg = cv2.pyrDown(img)
    hdx = cv2.Sobel(himg, cv2.CV_32F, 0, 1, ksize=3)
    hdy = cv2.Sobel(himg, cv2.CV_32F, 1, 0, ksize=3)
    hl = np.sqrt(hdx**2+hdy**2)
    hl = cv2.pyrUp(hl)
    l = l - hl'''

    '''
    img = cv2.medianBlur(img, 5)
    #img = cv2.boxFilter(img, cv2.CV_32F, (5,5))
    dx = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    l2 = np.sqrt(dx**2+dy**2)
    #l =  l2 * l
    '''

    l = (l>9).astype(np.uint8)

    l = skeletonize(l).astype(np.uint8)

    '''
    cimg,cntrs,h = cv2.findContours(l, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cntr_img = np.zeros( (img.shape[0],img.shape[1],3) , dtype=np.uint8)
    random.shuffle(cntrs)
    for i,cntr in enumerate(cntrs):
        print(cntr.shape,cntr)
        c = inferno(i/len(cntrs))
        c = tuple(int(a*255) for a in c[0:3])
        img = cv2.polylines(cntr_img, cntr[:], True, c)
    print('num cntrs',len(cntrs))
    cv2.imshow('cntr', cntr_img); cv2.waitKey(0); cv2.destroyAllWindows()
    '''

    ccs = measure.label(l, background=0)
    print('num ccs', ccs.max())
    # Delete small ccs.
    for cc in range(1,ccs.max()):
        inds = np.argwhere(ccs == cc).T
        if (inds.shape[1]) < 10:
            ccs[inds[0],inds[1]] = 0
    unique_ccs = list(np.unique(ccs))
    unique_ccs.remove(0)
    print('num ccs', len(unique_ccs))
    # Renumber
    for i,cc in enumerate(unique_ccs):
        ccs[ccs==cc] = i
    normalize_and_show(ccs)

    sites = np.argwhere(ccs>0).T
    regions = []
    for i in range(1,ccs.max()):
        a = np.stack(np.where(ccs==i)).mean(1)
        region = [*a, i, 0]
        regions.append(region)

    import triangle
    vertices = sites.T.tolist()
    #print(regions)
    tri = triangle.triangulate(dict(vertices=vertices, regions=regions))
    print(tri)
    verts = tri['vertices']
    dimg = np.copy(img)
    for i,v in enumerate(tri['triangles']):
        p = np.stack( (verts[v[0]],verts[v[1]],verts[v[2]]) , 0 )[np.newaxis]
        p = p.astype(np.int32)
        p[..., [0,1]] = p[..., [1,0]]
        c = img[p[0,0,1],p[0,0,0]]
        c += img[p[0,1,1],p[0,1,0]]
        c += img[p[0,2,1],p[0,2,0]]
        c = inferno(c/(img.max()*3))
        c = tuple(int(a*255) for a in c[0:3])
        cv2.fillPoly(dimg, p, c)
    img = dimg


    #l = ((l - l.min()) / (l.max()-l.min()) * 255).astype(np.uint8)
    #l = cv2.createCLAHE().apply(l)

    #img = l
    #img = ((img - img.min()) / (img.max()-img.min()) * 255).astype(np.uint8)
    #l = cv2.Canny(img, 30, 100)


    #img = cv2.GaussianBlur(img, (7,7), 4)
    #l = img * cv2.Laplacian(img, cv2.CV_32F)
    #l = abs(l)

    #normalize_and_show(l)
    normalize_and_show(img)




img = cv2.imread('data/depth0.png',0)
img = img.astype(np.float32)/255
img = img * 90 - 2
mesh_tiff(img)
