import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

class SingletonApp:
    _instance = None

    def __init__(self, wh, name='Viz'):
        SingletonApp._instance = self
        self.wh = wh
        self.window = None

        self.last_x, self.last_y = 0,0
        self.left_down, self.right_down = False, False
        self.scroll_down = False
        self.left_dx, self.left_dy = 0,0
        self.right_dx, self.right_dy = 0,0
        self.scroll_dx, self.scroll_dy = 0,0
        self.name = name
        self.pickedPointClipSpace = None

    def do_init(self):
        raise NotImplementedError('must implement')

    def render(self):
        raise NotImplementedError('must implement')

    def idle(self, rs):
        self.render(rs)

    def keyboard(self, k, *args):
        if k == b'q': sys.exit()

    def mouse(self, but, st, x,y):
        if but == GLUT_LEFT_BUTTON and (st == GLUT_DOWN):
            if not self.left_down: self.pick(x,y)
            self.last_x, self.last_y = x, y
            self.left_down = True
        else:
            self.pickedPointClipSpace = None
        if but == GLUT_LEFT_BUTTON and (st == GLUT_UP):
            self.left_down = False
        if but == GLUT_RIGHT_BUTTON and (st == GLUT_DOWN):
            self.last_x, self.last_y = x, y
            self.right_down = True
        if but == GLUT_RIGHT_BUTTON and (st == GLUT_UP):
            self.right_down = False
        if but == 3 and (st == GLUT_DOWN):
            self.scroll_dy = self.scroll_dy * .7 + .9 * (-1) * 1e-1
        if but == 4 and (st == GLUT_DOWN):
            self.scroll_dy = self.scroll_dy * .7 + .9 * (1) * 1e-1
    def motion(self, x, y):
        if self.left_down:
            self.left_dx = self.left_dx * .5 + .5 * (x-self.last_x) * 1e-1
            self.left_dy = self.left_dy * .5 + .5 * (y-self.last_y) * 1e-1
        if self.right_down:
            self.right_dx = self.right_dx * .5 + .5 * (x-self.last_x) * 1e-1
            self.right_dy = self.right_dy * .5 + .5 * (y-self.last_y) * 1e-1

        self.last_x, self.last_y = x,y

    def reshape(self, w,h):
        glViewport(0, 0, w, h)
        self.wh = w,h

    def _render(*args):
        glutSetWindow(SingletonApp._instance.window)
        SingletonApp._instance.render(*args)
    def _idle(*args):
        glutSetWindow(SingletonApp._instance.window)
        SingletonApp._instance.idle(*args)
    def _keyboard(*args): SingletonApp._instance.keyboard(*args)
    def _mouse(*args):
        SingletonApp._instance.mouse(*args)
    def _motion(*args): SingletonApp._instance.motion(*args)
    def _reshape(*args): SingletonApp._instance.reshape(*args)

    def init(self, init_glut=False):
        if init_glut:
            glutInit(sys.argv)
            glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)

        glutInitWindowSize(*self.wh)
        self.reshape(*self.wh)
        self.window = glutCreateWindow(self.name)
        #glutSetWindow(self.window)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glAlphaFunc(GL_GREATER, 0)
        glEnable(GL_ALPHA_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.do_init()
        glutReshapeFunc(SingletonApp._reshape)
        glutDisplayFunc(SingletonApp._render)
        glutIdleFunc(SingletonApp._idle)
        glutMouseFunc(SingletonApp._mouse)
        glutMotionFunc(SingletonApp._motion)
        glutKeyboardFunc(SingletonApp._keyboard)

    def run_glut_loop(self):
        glutMainLoop()

    def pick(self, x,y):
        y = self.wh[1] - y - 1
        z = float(glReadPixels(x,y, 1,1, GL_DEPTH_COMPONENT, GL_FLOAT).squeeze())
        x = 2 * x / self.wh[0] - 1
        #y = -(2 * y / self.wh[1] - 1)
        y = (2 * y / self.wh[1] - 1)
        self.pickedPointClipSpace = np.array((x,y,1)) * z


def look_at_z_forward(eye, center, up):
    #forward = -center + eye; forward /= np.linalg.norm(forward)
    forward = center - eye; forward /= np.linalg.norm(forward)
    side = np.cross(forward, up); side /= np.linalg.norm(side)
    up = np.cross(side, forward)
    m = np.eye(4, dtype=np.float32)
    m[0,:3] = side
    m[1,:3] = up
    m[2,:3] = forward
    mt = np.eye(4)
    mt[:3,3] = eye
    m = m @ mt
    return m
def frustum_z_forward(left, right, bottom, top, near, far):
    #left, right = right, left
    #top, bottom = bottom, top
    return np.array((
        (2*near/(right-left), 0, (right+left)/(right-left), 0),
        (0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0),
        (0, 0, (far+near)/(far-near), -2*far*near/(far-near)),
        (0,0,1.,0)), dtype=np.float32)

class OctreeApp(SingletonApp):
    def __init__(self, wh):
        super().__init__(wh)
        self.time = 0
        self.phi = np.pi/4
        self.lam = np.pi/4
        self.rad = 4
        self.center = np.array((.5,.5,0), dtype=np.float32)
    def render(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    def do_init(self):
        pass

    def updateCamera(self, dt):
        self.time += dt
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if False:
            D = 1 + np.sin(self.time/4)**2 * 2
            gluLookAt(np.cos(self.time)*D,np.sin(self.time)*D,1+D, 0,0,0, 0,0,1)
        else:
            self.phi += self.left_dx/10
            self.lam -= self.left_dy/10
            self.rad += self.right_dy
            x = np.sin(self.lam) * np.cos(self.phi) * self.rad
            y = np.sin(self.lam) * np.sin(self.phi) * self.rad
            z = np.cos(self.lam) * self.rad
            x,y,z = (x,y,z)+self.center
            gluLookAt(x,y,z, *self.center, 0,0,1)
            self.left_dx=self.left_dy=self.right_dy=0

        near = .02
        u = np.tan(np.deg2rad(35)) * near
        v = np.tan(np.deg2rad(35)) * near
        #m = frustum_z_forward(-u,u,-v,v,1,50)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-u,u, -v,v, near,50)
        #glLoadMatrixf(m)
