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
    img = np.array(dset.bboxNative(patchBbox, res,int(aspect_hw*res), True))
    #cv2.imshow('tex',tex);cv2.waitKey(0)
    return img
def make_tex_from_img(img):
    #img = np.eye(512, dtype=np.uint8) * 250
    #img = np.stack((img,img,img),-1)
    print(' - img', img.shape, img.dtype)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    img = np.copy(img,'C')
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1],img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    assert glGetError() == 0
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    assert glGetError() == 0
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

class Meshing():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.setdefault('iso', 1e-6)
        self.cfg.setdefault('meshRawPoints', False)
        self.cfg.setdefault('meshRBF', True)

    def run(self, meta):
        pts,vals = meta['pts'], meta['vals']
        loc = np.zeros(4,dtype=np.int32)

        if self.cfg['meshRawPoints']:
            st = time.time()
            tree = pycmeshedup.Octree(loc, self.cfg['maxDepth'])
            tree.addMany(0, pts, vals)
            print(' - Octree indexing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

            st = time.time()
            rawMesh = pycmeshedup.meshOctree(tree, self.cfg['iso'])
            print(' - Raw point octree meshing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

            if 'img' in meta:
                rawMesh.tex = make_tex_from_img(meta['img'])
                self.make_uvs(rawMesh, meta)

            st = time.time()
            pycmeshedup.computeVertexNormals(tree,rawMesh);
            print(' - Octree computeVertexNormals took {:.1f}ms'.format((time.time()-st)*1000))
            rawMesh.print()
            return tree, rawMesh, None

        elif self.cfg['meshRBF']:
            st = time.time()
            tree = pycmeshedup.Octree(loc, self.cfg['maxDepth'])
            tree.addMany(0, pts, vals)
            print(' - Octree indexing {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

            st = time.time()
            ptNormals = pycmeshedup.computePointSetNormals(tree, pts, pts)
            print(' - computePointSetNormals on {} pts took {:.1f}ms'.format(len(pts),(time.time()-st)*1000))

            st = time.time()
            negPts = pycmeshedup.getNegativePoints(tree, pts, ptNormals, .001)
            print(' - getNegativePoints generated {} pts in {:.1f}ms'.format(len(negPts),(time.time()-st)*1000))

            return tree, None, ptNormals

    def make_uvs(self, mesh, meta):
        uvs = np.copy(mesh.verts[:, :2], 'C')
        uvs[:,1] = 1 - uvs[:,1]
        mesh.uvs = uvs

def get_dc_lidar_data(cfg):
    import pylas, ctypes
    stride = 1
    #f = '/data/lidar/USGS_LPC_VA_Fairfax_County_2018_e1617n1924.laz'
    f,stride = '/data/lidar/dc1.las', 8
    #f,stride = '/data/lidar/PA_Statewide_S_2006-2008_002771.las', 2
    #f,stride = '/data/lidar/airport.las', 4

    cfg.setdefault('maxDepth', 12)

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

        maxEdge = max(x2-x1, y2-y1)
        #xx,yy,zz = (x-x1) / maxEdge, (y-y1) / maxEdge, (z-z1) / maxEdge

        st1 = time.time()
        #pts = np.stack((xx,yy,zz), -1).astype(np.float32)
        pts = ((np.stack((x,y,z), -1) - (x1,y1,z1)) / maxEdge).astype(np.float32)
        print('   - las stack    took {:.2f}ms'.format((time.time()-st1)*1000))
        size = 2
        print(pts[::pts.shape[0]//5])
    print(' - las total load took {:.2f}ms'.format((time.time()-st0)*1000))

    vals = np.ones_like(pts[:,0])

    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = (x1,y1,z1)
    M[:3,:3] = np.diag((maxEdge,)*3)

    endPoints = np.array(( (0,0,0,1.),
                           (1,1,1,1.))) @ M.T
    endPoints = endPoints[:, :3] / endPoints[:, 3:]
    utm_bbox = *endPoints[0,:2], *(endPoints[1,:2]-endPoints[0,:2])
    ox,oy = -2, -8
    utm_bbox = (utm_bbox[0]+ox,utm_bbox[1]+oy,utm_bbox[2],utm_bbox[3])
    img = get_tiff_patch('/data/dc_tiffs/dc.tif', utm_bbox, 2048+2048)

    return dict(
            pts=pts,
            vals=vals,
            grid2native=M,
            img=img,
            )

def get_simple_data(cfg):
    N = 16*2
    cfg.setdefault('maxDepth', 4)
    xyz = (np.linspace(0,1,N,False),)*3
    pts = np.stack(np.meshgrid(*xyz), -1).reshape(-1,3).astype(np.float32)
    pts += .5/N

    vals = np.linalg.norm(pts, axis=1).astype(np.float32)
    #vals = abs(pts).max(1).astype(np.float32)
    #vals = (abs(pts)**.5).sum(1).astype(np.float32)
    #vals = abs(1 - vals)
    vals = .8 - vals

    x,y,z = pts[:,0],pts[:,1],pts[:,2]
    (x1,x2),(y1,y2),(z1,z2) = np.quantile(x[::4],[.1,.9]), np.quantile(y[::4],[.1,.9]), np.quantile(z[::4],[.1,.9])
    modelScale = x2-x1
    M = np.eye(4, dtype=np.float32)
    return dict(
            pts=pts,
            vals=vals,
            grid2native=M,
            )



def render_loop(app, tree, mesh, pts=None, tri_mesh=None, vertNormalMesh=None, ptNormalMesh=None):

    glEnable(GL_CULL_FACE)
    #glDisable(GL_CULL_FACE)
    for i in range(100000):
        app.updateCamera(.01)
        app.render()

        glColor4f(0,0,1,.5)
        tree.render(2)

        glEnable(GL_BLEND)

        glBegin(GL_LINES)
        s = 2
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
            glColor4f(.5, .5, .8, .2)
            glPointSize(1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, pts)
            #glEnableClientState(GL_COLOR_ARRAY)
            #glColorPointer(4, GL_FLOAT, 0, colors)
            glDrawArrays(GL_POINTS, 0, len(pts))
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

        '''
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
        '''
        glColor4f(0,1,.3,.3)
        if tri_mesh is not None: tri_mesh.render()
        glColor4f(.2, .6, .9, .6)
        if vertNormalMesh is not None: vertNormalMesh.render()
        glColor4f(.6, .9, .2, .6)
        if ptNormalMesh is not None: ptNormalMesh.render()
        if mesh is not None: mesh.render()


        time.sleep(.008)
        glutSwapBuffers()
        glutPostRedisplay()
        glutMainLoopEvent()
        glFlush()
        #print(' - frame')


if __name__ == '__main__':
    app = OctreeApp((1000,1000))
    app.init(True)

    cfg = {}
    meta = get_dc_lidar_data(cfg)
    #meta = get_simple_data(cfg)

    meshing = Meshing(cfg)
    tree,mesh,ptNormals = meshing.run(meta)


    tri_mesh = None
    vertNormalMesh = None
    if mesh is not None:
        #tri_mesh = pycmeshedup.convertTriangleMeshToLines(mesh)
        vertNormalMesh = pycmeshedup.normalsToMesh(mesh.verts, mesh.vertexNormals, .002)

    if ptNormals is not None:
        ptNormalMesh = pycmeshedup.normalsToMesh(meta['pts'], ptNormals, .002)

    pts = None
    pts = meta['pts']
    render_loop(app, tree, mesh, tri_mesh=tri_mesh, vertNormalMesh=vertNormalMesh, pts=pts, ptNormalMesh=ptNormalMesh)
