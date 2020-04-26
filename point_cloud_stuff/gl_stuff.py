import numpy as np
import time, sys
import sys, cv2
from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.GL.shaders

'''
I just copied this from another project of mine.
There's a ton of stuff I'm not going to use, but idm
'''

def homogeneousize(a):
    if a.ndim == 1: return np.array( (*a,1) , dtype=a.dtype )
    return np.stack( (a,np.ones((a.shape[0],1),dtype=a.dtype)) )

def make_4x4f(a):
    if a.shape != (4,4):
        b = np.eye(4, dtype=np.float32)
        b[:a.shape[0], :a.shape[1]] = a
        return b
    return a.astype(np.float32)

def ray_plane_xsect(p0, dir, q, n):
    n, dir = n / np.linalg.norm(n), dir / np.linalg.norm(dir)
    t = ((q - p0) @ n) / (dir @ n)
    return p0 + dir * t

def compile_and_link(vsrc, fsrc):
    vs = OpenGL.GL.shaders.compileShader(vsrc, GL_VERTEX_SHADER)
    fs = OpenGL.GL.shaders.compileShader(fsrc, GL_FRAGMENT_SHADER)
    return OpenGL.GL.shaders.compileProgram(vs, fs)

class ShaderUsage:
    def __init__(self, shader):
        self.shader = shader
        self.old = glGetIntegerv(GL_CURRENT_PROGRAM)
        glUseProgram(self.shader.prog)
        self.used = set([])
    def setAttrib(self, name, v):
        try:
            self.shader.setAttrib(name, v)
            self.used.add(name)
        except:
            print(' - failed to set attrib', name)
    def setUniform(self, name, v):
        try:
            self.shader.setUniform(name, v)
            self.used.add(name)
        except:
            print(' - failed to set uniform', name)
    def cleanup(self):
        #assert(all([attrib in self.used for attrib in self.shader.allAttribs]))
        #assert(all([uni in self.used for uni in self.shader.allUniforms]))
        for attrib,attrib_id in self.shader.allAttribs.items():
            self.shader.unsetAttrib(attrib_id)
        glUseProgram(self.old)

class Shader:
    def __init__(self, name, vsrc, fsrc):
        self.name = name
        vs = OpenGL.GL.shaders.compileShader(vsrc, GL_VERTEX_SHADER)
        fs = OpenGL.GL.shaders.compileShader(fsrc, GL_FRAGMENT_SHADER)
        self.prog = OpenGL.GL.shaders.compileProgram(vs, fs)
        self.cur_usage = []
        assert(self.prog)

        if True:
            glUseProgram(self.prog)
            nUniforms = glGetProgramiv(self.prog, GL_ACTIVE_UNIFORMS)
            nAttributes = glGetProgramiv(self.prog, GL_ACTIVE_ATTRIBUTES)
            allAttributes, allUniforms = {}, {}
            for i in range(nUniforms):
                name, size, type = glGetActiveUniform(self.prog, i)
                name = name.decode('ascii')
                allUniforms[name] = i
            for i in range(nAttributes):
                l,s,t,n = '',bytes([0]*32),bytes([0]*32),bytes([0]*32)
                r = glGetActiveAttrib(self.prog, i, 32,l,s,t,n)
                n = n.decode('ascii').strip('\t\r\n\0')
                loc = glGetAttribLocation(self.prog, n)
                allAttributes[n] = loc
        self.allAttribs = allAttributes
        self.allUniforms = allUniforms
        print(' - attribs:\n', self.allAttribs)
        print(' - uniforms:\n', self.allUniforms)

    def __enter__(self):
        self.cur_usage.append(ShaderUsage(self))
        return self.cur_usage[-1]
    def __exit__(self, *a):
        u = self.cur_usage.pop()
        u.cleanup()

    def setUniform(self, name, val):
        i = self.allUniforms[name]
        #print('set uni', name, i)
        if type(val) == int:
            glUniform1i(i, val)
        elif type(val) == float:
            glUniform1f(i, val)
        elif val.shape == (1,) and val.dtype==int:
            glUniform1i(i, val)
        elif val.shape == (3,):
            glUniform3f(i, *val)
        elif val.shape == (4,):
            glUniform4f(i, *val)
        elif val.shape == (4,4):
            glUniformMatrix4fv(i, 1,False, val.T) # note: transpose!
        else:
            assert(False)

    def setAttrib(self, name, array):
        glEnableVertexAttribArray(self.allAttribs[name])
        glVertexAttribPointer(self.allAttribs[name], array.shape[1], GL_FLOAT, GL_FALSE, 0, array)
        #print('set attrib', name, self.allAttribs[name], array)

    def unsetAttrib(self, name_or_id):
        if isinstance(name_or_id, str):
            glDisableVertexAttribArray(self.allAttribs[name_or_id])
        else: glDisableVertexAttribArray(name_or_id)


def make_shaders():
    basicColored = Shader('basicColored', '''
            #version 400
            uniform mat4 mvp;
            in layout(location = 0) vec3 a_position;
            in layout(location = 1) vec4 a_color;
            out vec4 v_color;
            void main() {
                gl_Position = mvp * vec4(a_position, 1.0);
                v_color = a_color;
            } ''', '''
            #version 400
            precision mediump float;
            in vec4 v_color;
            void main() {
                gl_FragColor = v_color;
            }''')

    basicTextured = Shader('texturedShader', '''
        #version 400
        in layout(location = 0) vec3 a_position;
        in layout(location = 1) vec2 a_uv;
        uniform mat4 mvp;
        out vec2 v_uv;
        void main() {
            gl_Position = mvp * vec4(a_position, 1.0);
            //v_uv = a_uv * .001 + a_position.xy;
            v_uv = a_uv;
        }''', '''
        #version 400
        in vec2 v_uv;
        uniform sampler2D tex;
        uniform vec4 u_color;
        void main() {
            gl_FragColor = vec4(texture2D(tex, v_uv).rgba) * u_color;
            //gl_FragColor += vec4(1.0);
        }''')

    '''
    See this:
        http://www.reedbeta.com/blog/quadrilateral-interpolation-part-1/
    '''
    perspectiveTextured = Shader('perspectiveTextured', '''
            #version 400
            uniform mat4 mvp;
            in layout(location = 0) vec3 a_position;
            in layout(location = 1) vec3 a_uvq;
            out vec3 v_uvq;
            void main() {
                gl_Position = mvp * vec4(a_position, 1.0);
                v_uvq = a_uvq;
            } ''',  '''
            #version 400
            precision mediump float;
            uniform sampler2D sampler;
            uniform vec4 u_color;
            in vec3 v_uvq;
            void main() {
                vec2 uv = v_uvq.xy / v_uvq.z;
                vec4 c = texture2D(sampler, uv ) * u_color;
                gl_FragColor = c;
            }''')

    projectedCaster = Shader('projectedCaster', '''
            #version 400
            uniform mat4 mvp;
            uniform mat4 caster_mvp;

            in layout(location = 0) vec3 a_position;
            out vec4 v_casted_xyzw;

            void main() {
                gl_Position = mvp * vec4(a_position, 1.0);
                v_casted_xyzw = caster_mvp * vec4(a_position, 1.0);
            } ''',  '''
            #version 400
            precision mediump float;
            uniform sampler2D sampler;
            in vec4 v_casted_xyzw;
            void main() {
                vec2 casted_uv = v_casted_xyzw.xy / v_casted_xyzw.z;
                vec4 c = texture2D(sampler, casted_uv );
                c.r = 1.0;
                c.a = .5;
                gl_FragColor = c;
            }''')

    return {
        'basicColored': basicColored,
        'basicTextured': basicTextured,
        'perspectiveTextured': perspectiveTextured,
        'projectedCaster': projectedCaster,
        }

class RenderState:
    def __init__(self, view, proj, shaders={}, overrideShader=None):
        self.view = view
        self.proj = proj
        self.mvp = proj @ view

        self.shaders = shaders
        self.overrideShader = overrideShader

    def getShader(self, s):
        if self.overrideShader: return self.overrideShader
        return self.shaders[s]

    def copy(self):
        rs = RenderState(self.view, self.proj, shaders=self.shaders)
        return rs

class Compass:
    def __init__(self, width=2, scale=10, pos=np.zeros(3)):
        self.width = width
        self.scale = scale
        self.color = np.array((
            (1,0,0,1), (1,0,0,1),
            (0,1,0,1), (0,1,0,1),
            (0,0,1,1), (0,0,1,1.)), dtype=np.float32)
        self.pos = np.array((
            (scale,0,0), (0,0,0),
            (0,scale,0), (0,0,0),
            (0,0,scale), (0,0,0.)), dtype=np.float32) + pos[np.newaxis].astype(np.float32)*0
    def draw(self, rs):
        glLineWidth(self.width)
        with rs.getShader('basicColored') as shader:
            shader.setAttrib('a_color', self.color)
            shader.setAttrib('a_position', self.pos)
            shader.setUniform('mvp', rs.mvp)
            glDrawArrays(GL_LINES, 0, 6)

'''
Default glu functions use Z- as forward, but this is at odds with
most computer vision work and general intuition that Z+ is pointing in front of the camera.
So I prefer Z+ foward, Y+ down. This has the same chirality since we flip two axes.
'''
def ortho_z_forward(left, right, bottom, top, near, far):
    return np.array((
        (2/(right-left), 0, 0, 0),
        (0, 2/(top-bottom), 0, 0),
        (0, 0, 2/(far-near), 0),
        (-(right+left)/(right-left), -(top+bottom)/(top-bottom), -(far+near)/(far-near), 1)), dtype=np.float32).T
def frustum_z_forward(left, right, bottom, top, near, far):
    #left, right = right, left
    #top, bottom = bottom, top
    return np.array((
        (2*near/(right-left), 0, (right+left)/(right-left), 0),
        (0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0),
        (0, 0, -(far+near)/(far-near), -2*far*near/(far-near)),
        (0,0,-1.,0)), dtype=np.float32)
    return np.array((
        (2*near/(right-left), 0, (right+left)/(right-left), 0),
        (0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0),
        (0, 0, (far+near)/(far-near), -2*far*near/(far-near)),
        (0,0,1.,0)), dtype=np.float32)
def look_at_z_forward(eye, center, up):
    #forward = -center + eye; forward /= np.linalg.norm(forward)
    forward = center - eye; forward /= np.linalg.norm(forward)
    side = np.cross(forward, up); side /= np.linalg.norm(side)
    up = np.cross(side, forward)
    m = np.eye(4, dtype=np.float32)
    m[0,:3] = side
    m[1,:3] = up
    m[2,:3] = -forward
    mt = np.eye(4)
    mt[:3,3] = -eye
    m = m @ mt
    return m

class Camera:
    # If viewAndProjection is passed, must be tuple of 2 4x4 matrices.
    # Otherwise, must specify eye/center/up
    def __init__(self, eye=None, center=None, up=None, viewAndProjection=None, fov=np.deg2rad(35), aspect_hw=1, z_near=8):
        if viewAndProjection is None:
            self.view = look_at_z_forward(eye,center,up)

            #self.proj = ortho_z_forward(-1,1,-1,1,-10,10)
            zz = z_near # near plane dist.
            hfov = fov / aspect_hw
            vfov = fov
            u = np.tan(hfov * .5) * zz
            v = np.tan(vfov * .5) * zz
            self.proj = frustum_z_forward(-u,u,-v,v,zz,50000)
        else:
            self.view = viewAndProjection[0]
            self.proj = viewAndProjection[1]
            eye = np.linalg.inv(self.view)[:3,3]

        self.t = eye
        self.R = self.view[:3,:3]

    def update_arcball(self, anchor, d):
        if np.linalg.norm(d) < .0001: return

        tt = self.t
        rr0 = self.R
        rr = rr0

        dr = np.copy(d) / 500
        dr[2] = 0
        dr[[0,1]] = dr[[1,0]]
        inc_ = cv2.Rodrigues(rr @ dr )[0]
        inc = np.eye(4)
        inc[:3,:3] = inc_
        print('inc',inc)

        P = np.eye(4); P[:3,:3] = rr; P[:3,3] = tt
        P[:3,3] -= anchor
        P[:3,3] *= (1+d[2]/500)
        P = np.linalg.inv(P)
        P = P @ inc
        P = np.linalg.inv(P)
        P[:3,3] += anchor
        self.R = P[:3,:3]
        self.t = P[:3,3]
        P = np.linalg.inv(P)
        self.view = P


        '''
        dr = np.copy(d)
        dr[2] *= 0
        dr[[0,1]] = dr[[1,0]]
        dr[1] *= -1
        dr = rr.T @ dr / 500
        inc =cv2.Rodrigues(rr @ dr )[0]
        rr = inc @ rr

        #tt += rr @ d
        scroll_dir = anchor - tt
        scroll_dir = scroll_dir / (np.linalg.norm(scroll_dir)+1e-6)
        print(scroll_dir)
        #tt += scroll_dir * d[2]

        #tt = tt*.4 + inc@tt*.6
        tt = tt - anchor + inc @ (anchor - tt)
        print('self.t',self.t)
        print('tt',tt)
        self.view = np.eye(4)
        self.view[:3,:3] = rr.T
        self.view[:3,3] = -rr.T@tt
        self.R = rr
        self.t = tt
        '''

        pass


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

    def keyboard(self, *args):
        sys.exit()

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
            glutInitDisplayMode(GLUT_RGB)

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
