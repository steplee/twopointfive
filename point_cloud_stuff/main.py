import numpy as np
import pylas
import ctypes, cv2

from .gl_stuff import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lidarFile', default='data/USGS_LPC_MD_VA_Sandy_NCR_2014_18SUJ321308_LAS_2015.laz')
parser.add_argument('--tiff', default='/home/slee/stuff/terrapixel/terraslam/data/dc/dc.tiff')
parser.add_argument('--N', default=90000000, type=int)
args = parser.parse_args()


if args.tiff == None or len(args.tiff) < 2:
    dset = None
else:
    # TODO: Remove dependency on my code for work.
    from pytavio.geodata.gdal_dataset import GeoDataset
    dset = GeoDataset(args.tiff)

with pylas.open(args.lidarFile) as fh:
    print('Points from Header:', fh.header.point_count)
    las = fh.read()
    print(las)
    print('Points from data:', len(las.points))
    print(las.x)
    print(las.y)
    print(las.z)
    print(las.points_data.point_format.dimension_names)

    N = args.N
    s = 1
    x,y,z = las.x[0:N:s],las.y[:N:s],las.z[:N:s]
    n = len(x)

    if dset is None:
        from matplotlib.cm import inferno
        inten = las.intensity.astype(np.float32)[0:N:s]
        div = min(inten.max(), inten.mean()*2.5)
        inten = (inten/div)
        color = inferno(inten).astype(np.float32) # Looks pretty cool
    else:
        x1,y1,x2,y2 = x.min(), y.min(), x.max(), y.max()
        xywh = xxyy = x1,y1, x2-x1, y2-y1
        print('xxyy', x1,y1,x2,y2)
        print('xywh', xywh)
        pix_bb = tuple(int(a) for a in dset.xform_bbox_utm2pix(xywh))
        print('pix_bb', pix_bb)
        img = dset.bbox(*pix_bb, int(xywh[2]), int(xywh[3])) # Sample at 1 px/m
        samples = np.stack((y,x),1) - (y1,x1)
        OFFSET_X, OFFSET_Y = 0, 10
        # Flip (UTM is north up, image is up down)
        samples[:,0] = (img.shape[0] - 1+OFFSET_Y - samples[:,0]).clip(min=0,max=img.shape[0]-1)
        #samples[:,1] = img.shape[1] - samples[:,1].clip(min=0,max=img.shape[1]-1) - 1
        samples[:,1] = (samples[:,1]+OFFSET_X).clip(min=0,max=img.shape[1]-1)
        samples = samples.astype(np.int32)
        print('SAMPLES',samples)
        color = img[samples[:,0], samples[:,1]]
        color = color.astype(np.float32) / 255.
        color = np.hstack( (color,np.ones((len(color),1),dtype=np.float32)) )

    pts = np.stack((x,y,z), -1).astype(np.float32)
    pts = pts - pts.mean(0)
    pts = pts / np.quantile(abs(pts), .96)
    pts = pts - pts.mean(0)
    pts = pts * 50000
    pts = pts - pts.mean(0)
    pts = pts - pts.mean(0)
    print('max/min',pts.max(0), pts.min(0))

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import time

class LasApp(SingletonApp):
    def __init__(self, wh, cloud, color=None):
        super().__init__(wh)
        self.cloud = cloud
        #color = np.stack((np.sin(cloud[:, 0]), np.cos(cloud[:,1]), np.sin(cloud[:,2]), np.ones(len(cloud))),-1).astype(np.float32)
        self.color = np.ones( (cloud.shape[0],4), dtype=np.float32 ) if color is None else color

        self.eye = np.zeros(3)
        self.eyeOff = np.array((-20,-20,300.))
        self.center0 = np.zeros(3)
        self.anchor = np.zeros(3)
        self.camera = None

    def do_init(self):
        self.compass = Compass(scale=200)
        self.shaders = make_shaders()

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        verts = np.hstack((self.cloud, self.color))
        print('arr size', verts.size, '*4')
        glBufferData(GL_ARRAY_BUFFER, verts.size*4, verts, GL_STATIC_DRAW)
        self.N = len(verts)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


    def render(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rs = self.rs

        if self.anchor is not None:
            with rs.getShader('basicColored') as shader:
                glPointSize(20)
                color = np.array( ((0,0,1.,1.),) , dtype=np.float32 )
                pos = self.anchor.astype(np.float32)[np.newaxis]
                shader.setAttrib('a_color', color)
                shader.setAttrib('a_position', pos)
                shader.setUniform('mvp', rs.mvp)
                glDrawArrays(GL_POINTS, 0, 1)


        glPointSize(3)

        self.compass.draw(rs)

        with rs.getShader('basicColored') as shader:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            pi,ci = shader.shader.allAttribs['a_position'], shader.shader.allAttribs['a_color']

            glEnableVertexAttribArray(pi)
            glEnableVertexAttribArray(ci)
            glVertexAttribPointer(pi, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
            glVertexAttribPointer(ci, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
            shader.setUniform('mvp', rs.mvp)
            glDrawArrays(GL_POINTS, 0, self.N)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        '''
        buf = glReadPixels(0,0, *self.wh, GL_RGB, GL_UNSIGNED_BYTE)
        buf = np.frombuffer(buf, dtype=np.uint8).reshape(*self.wh[::-1],3)
        buf = buf + 2
        import torch
        a = torch.from_numpy(buf).to(torch.float32).permute(2,0,1).unsqueeze_(0)
        b = torch.nn.functional.max_pool2d(a,(8,8))
        b = torch.nn.functional.interpolate(b, a.size()[2:4])
        c = torch.stack( (a,b) ).max(dim=0)[0]
        buf = c[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

        buf = buf[::-1]
        cv2.imshow('buf', cv2.cvtColor(buf,cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        '''

    def updateCamera(self, dt):
        if hasattr(self, 'rs') and self.pickedPointClipSpace is not None:
            if self.pickedPointClipSpace[2] > .999999:
                picked = np.array((0,0,0))
            else: picked = self.pickedPointClipSpace
            ws = np.linalg.inv(self.rs.proj) @ homogeneousize(picked)
            ws /= ws[3]
            ws *= .5
            viewi = np.linalg.inv(self.rs.view)
            anchor = viewi[:3,:3] @ ws[:3] + viewi[:3,3]
            #mvp_inv = np.linalg.inv(self.rs.mvp)
            #anchor = mvp_inv @ homogeneousize(self.pickedPointClipSpace)
            #anchor = anchor[:3] / anchor[3]
            self.anchor = anchor
            print('anchor',anchor)
            self.pickedPointClipSpace = None

        if self.camera is None:
            speed = np.linalg.norm(self.eye+self.eyeOff-self.center0) / 10 + .1
            scroll_speed = (speed) * 3 + .4
            self.eyeOff += np.array((-self.left_dx*speed, self.left_dy*speed, self.scroll_dy*scroll_speed))
            self.camera = Camera(
                    eye=self.eyeOff+self.eye,
                    center=self.center0-self.center0,
                    up=np.array((0,1,0)),
                    aspect_hw=self.wh[1]/self.wh[0],
                    z_near = 1
                    )

        if self.anchor is not None:
            speed = np.linalg.norm(self.eye+self.eyeOff-self.center0) / 10 + .1
            scroll_speed = (speed) * 3 + .4
            d = np.array((-self.left_dx*speed, self.left_dy*speed, self.scroll_dy*scroll_speed))
            self.camera.update_arcball(self.anchor, d)

        self.left_dy -= self.left_dy * dt * 25
        self.left_dx -= self.left_dx * dt * 25
        self.right_dy -= self.right_dy * dt * 25
        self.scroll_dy -= self.scroll_dy * dt * 25

        self.rs = RenderState(self.camera.view, self.camera.proj, self.shaders)


app = LasApp((1700,900), pts, color)
app.init(True)

for i in range(100000):
    app.updateCamera(.01)
    app.render()

    glEnable(GL_BLEND)

    glutMainLoopEvent()
    glFlush()
    glutSwapBuffers()
    time.sleep(.01)

