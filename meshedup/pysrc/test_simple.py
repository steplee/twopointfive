import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2

from OpenGL.GL import *
from OpenGL.GLUT import *
from .gl_stuff import *
import pycmeshedup

def test():
    if True:
        N = 822900
        pts = (np.random.uniform(-1,1, size=(N,3))).astype(np.float32)

        N = 16*1
        xyz = (np.linspace(-1,1,N,False),)*3
        pts = np.stack(np.meshgrid(*xyz), -1).reshape(-1,3).astype(np.float32)
        pts += 1/N

        #pts = (np.random.randn(N,3)/2).clip(-2,2).astype(np.float32)
        #vals = abs(np.random.randn(N)).clip(0,1).astype(np.float32)
        #vals = np.ones(N, dtype=np.float32)
        vals = np.linalg.norm(pts, axis=1).astype(np.float32)
        #vals = abs(pts).max(1).astype(np.float32)
        vals = abs(1 - vals)
        #vals = 1 - vals
    else:
        N = 9
        xyz = (np.linspace(-1,1,N,False),)*3
        pts = np.stack(np.meshgrid(*xyz), -1).reshape(-1,3).astype(np.float32)
        pts += .01
        #vals = np.linalg.norm(pts, axis=1).astype(np.float32)
        vals = abs(pts).sum(1).astype(np.float32)
        vals = vals - 1.0001


    size = 2
    tree = pycmeshedup.Octree(np.zeros(3,dtype=np.float32)-size/2, size, 0, 3)
    for id,(pt,val) in enumerate(zip(pts,vals)): tree.add(pt, id, val)

    iso = .2
    st = time.time()
    tris = pycmeshedup.meshOctree(tree, iso)
    tris = np.stack(tris)
    print(' - Octree indexing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))
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

    return tree, pts,vals,tris

tree, pts, vals, tris = test()

print('\n\n - SearchingNode.')
node = tree.searchNode(pts[1])
print(node.info())
del node


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


app = OctreeApp((1000,1000))
app.init(True)
glDisable(GL_CULL_FACE)
for i in range(100000):
    app.updateCamera(.01)
    app.render()

    glColor4f(0,0,1,.5)
    tree.render(99)

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

    if True:
        glColor4f(1,1,1,.5)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        glDrawArrays(GL_POINTS, 0, len(pts))
        glDisableClientState(GL_VERTEX_ARRAY)

    glColor4f(0,1,.3,.7)
    glPointSize(1)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, tris)
    glDrawArrays(GL_TRIANGLES, 0, len(tris)*3)
    glDisableClientState(GL_VERTEX_ARRAY)


    glutMainLoopEvent()
    glFlush()
    glutSwapBuffers()
    time.sleep(.007)
    #print(' - frame')

