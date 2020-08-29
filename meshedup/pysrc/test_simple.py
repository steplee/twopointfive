import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2

from OpenGL.GL import *
from OpenGL.GLUT import *
from .gl_stuff import *
import pycmeshedup

def testLidar():
    import pylas, ctypes
    stride = 1
    #f = '/data/lidar/USGS_LPC_VA_Fairfax_County_2018_e1617n1924.laz'
    f,stride = '/data/lidar/dc1.las', 2
    #f,stride = '/data/lidar/PA_Statewide_S_2006-2008_002771.las', 2
    #f,stride = '/data/lidar/airport.las', 4
    with pylas.open(f) as fh:
        N = -1
        las = fh.read()
        x,y,z = las.x[0:N:stride],las.y[:N:stride],las.z[:N:stride]
        #x1,y1,x2,y2,z1,z2 = x.min(), y.min(), x.max(), y.max(), z.min(), z.max()
        (x1,x2),(y1,y2),(z1,z2) = np.quantile(x[::4],[.1,.9]), np.quantile(y[::4],[.1,.9]), np.quantile(z[::4],[.1,.9])
        #xx,yy,zz = x - (x1+x2)/2, y - (y1+y2)/2, z - (z1+z2)/2
        modelScale = x2-x1
        xx,yy,zz = x - x1 - modelScale/2, y - y1 - modelScale/2, z - z1
        #x1,y1,x2,y2,z1,z2 = x.min(), y.min(), x.max(), y.max(), z.min(), z.max()
        #xq = np.quantile(abs(xx[::4]),.9)

        pts = np.stack((xx,yy,zz), -1).astype(np.float32)
        pts = pts / modelScale
        size = 2
        print(' - size', size)
        offset = np.ones(3,dtype=np.float32) * -size/2
        print(pts[::100000])

    vals = np.ones_like(pts[:,0])

    loc = np.zeros(4,dtype=np.int32)
    st = time.time()
    tree = pycmeshedup.Octree(offset, loc, size, 0, 10)
    id = 0
    '''
    pts2 = np.tile(pts.T,2).T + np.random.randn(pts.shape[0]*2,pts.shape[1]) / 30
    for pt in pts2:
        tree.add(pt, id, 0)
        id += 1
    '''
    for (pt,val) in (zip(pts,vals)):
        tree.add(pt, id, val)
        id += 1
    print(' - Octree indexing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

    iso = .0002
    colors = np.ones((len(vals),4), dtype=np.float32)
    colors[vals<iso] = (.9,.1,.2,.5)
    colors[vals>iso] = (.1,.9,.2,.5)

    import tview
    dset = tview.GeoDataset('/data/dc_tiffs/dc.tif')
    #bbox = np.array((x.min(),y.min(), x.max()-x.min(), y.max()-y.min())).astype(np.float64)
    bbox = np.array((x1,y1,x2-x1,y2-y1)).astype(np.float64)
    aspect_wh = bbox[3] / bbox[2]
    print(' - Dset bbox:', bbox)
    tex = np.array(dset.bboxNative(bbox, 2048,int(aspect_wh*2048), True))
    print(' - img shape', tex.shape)
    #cv2.imshow('tex',tex);cv2.waitKey(0)
    uvs = None



    st = time.time()
    tris = pycmeshedup.meshOctree(tree, iso)
    print(' - Octree meshing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))
    if len(tris)>0: tris = np.concatenate(tris,0)

    coords = (x1,x2, y1,y2, z1,z2, modelScale)
    return tree, pts,vals,tris, colors, uvs,tex, coords

def test1():
    if False:
        N = 1000
        #N = 120
        pts = (np.random.uniform(-1,1, size=(N,3))).astype(np.float32)
    else:
        N = 16*2
        N = 16*2
        xyz = (np.linspace(-1,1,N,False),)*3
        pts = np.stack(np.meshgrid(*xyz), -1).reshape(-1,3).astype(np.float32)
        pts += 1/N

    if True:
        np.random.shuffle(pts)
        pts = pts[:pts.shape[0]//2]

    #pts = (np.random.randn(N,3)/2).clip(-2,2).astype(np.float32)
    #vals = abs(np.random.randn(N)).clip(0,1).astype(np.float32)
    #vals = np.ones(N, dtype=np.float32)
    #vals = np.linalg.norm(pts, axis=1).astype(np.float32)
    vals = abs(pts).max(1).astype(np.float32)
    #vals = (abs(pts)**.5).sum(1).astype(np.float32)
    vals = abs(1 - vals)
    #vals = 1 - vals


    size = 2
    loc = np.zeros(4,dtype=np.int32)
    st = time.time()
    tree = pycmeshedup.Octree(np.zeros(3,dtype=np.float32)-size/2, loc, size, 0, 4)
    for id,(pt,val) in enumerate(zip(pts,vals)): tree.add(pt, id, val)
    print(' - Octree indexing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

    iso = .2
    colors = np.ones((len(vals),4), dtype=np.float32)
    colors[vals<iso] = (.9,.1,.2,.5)
    colors[vals>iso] = (.1,.9,.2,.5)

    st = time.time()
    tris = pycmeshedup.meshOctree(tree, iso)
    print(' - Octree meshing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))
    if len(tris)>0: tris = np.concatenate(tris,0)
    #for tri in tris: print(tri)

    '''print(' - Root info:\n', tree.info())
    print(' - Root->0 info:\n', tree.child(0).info())
    print(' - Root->0->0 info:\n', tree.child(0).child(0).info())
    print(' - Root->0->0->0 info:\n', tree.child(0).child(0).child(0).info())
    print(' - Root->1 info:\n', tree.child(1).info())
    print(' - Root->2 info:\n', tree.child(2).info())
    print(' - Root->3 info:\n', tree.child(3).info())
    print(' - Root->4 info:\n', tree.child(4).info())
    print(' - Root->5 info:\n', tree.child(5).info())'''

    '''
    with open('out/mesh.obj','w') as fp:
        for tri in tris:
            for pt in tri:
                fp.write('v {} {} {}\n'.format(*pt))
        ind = 1
        for tri in tris:
            fp.write('f {} {} {}\n'.format(ind,ind+1,ind+2))
            ind += 3
    '''

    uvs = tex = None
    return tree, pts,vals,tris, colors, uvs,tex, coords

app = OctreeApp((1000,1000))
app.init(True)

#tree, pts, vals, tris, colors = test1()
tree, pts, vals, tris, colors, uvs,img, (x1,x2,y1,y2,z1,z2,modelScale) = testLidar()
if img is not None:
    #uvs = tris[:,:2] - tris[:,:2].min(0)[np.newaxis]
    #uvs = uvs / uvs.max(0)[np.newaxis]
    #print('xy',x1,x2,y1,y2, 'sz', x2-x1,y2-y1)
    #uvs = (tris[:, :2] + (.5,.5)) * modelScale
    #uvs = uvs / (x2-x1,y2-y1)
    uvs = (tris[:, :2] + (1,1)) / 2
    #uvs[[0,1]] = uvs[[1,0]]
    #uvs[:,0] = 1 - uvs[:,0]
    #uvs[:,1] = 1 - uvs[:,1]
    print('UVS',uvs)


if img is not None:
    tex = glGenTextures(1)
    assert glGetError() == 0
    glBindTexture(GL_TEXTURE_2D, tex)
    assert glGetError() == 0
    img = np.copy(img,'C')
    assert glGetError() == 0
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1],img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    assert glGetError() == 0
    print(' - tex', tex)
else:
    tex = None


#print('\n\n - SearchingNode.')
#node = tree.searchNode(pts[1])
#print(node.info())
#del node

#tree.print(0)

print('\n\n - Searching.')
#nearest = tree.search(pts, pts[0], 2)
#print(' - Done Searching:', nearest)
st = time.time()
qpts = pts[np.random.choice(range(len(pts)),1000)]
#qpts = pts[:100]
d,i = tree.search(pts, qpts, 5)
dt = (time.time() - st) * 1000
print(' - Done Searching:')
print(' - Dists:\n', d)
print(' - Inds:\n', i)
print(' - Time to search {} needles from {} haystack: {:.2f}ms'.format(len(d),len(pts),dt))
print("\n\n")

if len(tris) > 1:
    tri_inds = []
    for i in range(0, len(tris), 3):
        tri_inds.extend([i,i+1, i,i+2, i+1,i+2])
    tri_inds = np.array(tri_inds,dtype=np.uint32)
else:
    tri_inds = []

glEnable(GL_CULL_FACE)
#glDisable(GL_CULL_FACE)
for i in range(100000):
    app.updateCamera(.01)
    app.render()

    glColor4f(0,0,1,.5)
    #tree.render(99)

    glEnable(GL_BLEND)

    glBegin(GL_LINES)
    s = 3
    glColor4f(1,0,0,0)
    glVertex3f(0,0,0)
    glColor4f(1,0,0,1)
    glVertex3f(s,0,0)
    glColor4f(0,1,0,0)
    glVertex3f(0,0,0)
    glColor4f(0,1,0,1)
    glVertex3f(0,s,0)
    glColor4f(0,0,1,0)
    glVertex3f(0,0,1)
    glColor4f(0,0,1,1)
    glVertex3f(0,0,s)
    glEnd()

    if False:
        glPointSize(1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(pts))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, tris)
    if tex is None:
        glColor4f(0,1,.3,.3)
        glDrawArrays(GL_TRIANGLES, 0, len(tris)*1)
    else:
        glColor4f(1,1,1,.9)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_FLOAT, 0, uvs)
        glDrawArrays(GL_TRIANGLES, 0, len(tris)*1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    if False:
        glLineWidth(2)
        glColor4f(0,1,.8,.4)
        glDrawElements(GL_LINES, len(tri_inds), GL_UNSIGNED_INT, tri_inds)
        glDisableClientState(GL_VERTEX_ARRAY)
        glLineWidth(1)


    time.sleep(.008)
    glutSwapBuffers()
    glutPostRedisplay()
    glutMainLoopEvent()
    glFlush()
    #print(' - frame')

