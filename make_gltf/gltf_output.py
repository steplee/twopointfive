import os, sys, numpy as np
import re
import json
import cv2
from OpenGL.GL import *

'''
For now, we use 'pushPrimitivePart()' to build our mesh from any number of elements.
In 'finish()', the class will aggregate functionally-equiavlent 'primitive parts' and
write the mesh to GltfWriter using 'putMesh()'. From here this class is done -- the user
should use the returned mesh_id from 'finish()' to add a node, then a node to a scene.
'''
class OutputMesh:
    def __init__(self, name):
        # Our simple technique is to keep one primitive per attr/mode, and just add to it.
        #self.primitives = {}
        self.primitives = []
        self.finished = False
        self.name = name

    '''
    # here we push to a dict-of-lists, but it will be flattened later.
    def pushPrimitivePart(self, attributes, indices, mode, material=0):
        assert(not self.finished)
        attribues = sorted(list(attributes.items()))
        hashed = hash(str(attributes)+str(mode)+str(material))
        if hashed not in self.primitives:
            self.primitives[hashed] = []

        self.primitives[hashed].append(dict(attributes=attributes,
            indices=indices,mode=mode,material=material))
    '''

    def pushPrimitive(self, attributes, indices, mode, material=0):
        self.primitives.append(dict(attributes=attributes, indices=indices, mode=mode, material=material))


    def finish(self, writer):
        # Take our primitives and concatenate each inner list.
        assert(not self.finished)
        out_prims = []
        print(' - Finishing Mesh:')
        for prim in self.primitives:
            attributes = list(prim['attributes'].items())
            attribute_keys = [a[0] for a in attributes]
            attribute_data = [a[1] for a in attributes]
            adata = np.hstack(attribute_data)
            print('ADATA', adata.shape, adata.dtype)

            num_attrs = len(attribute_keys)
            num_verts = adata.shape[0]
            num_elems = adata.shape[1]
            byteStride = num_elems * adata.itemsize
            print('\t- Primitive with ({} attributes, {} elems, {} stride, {} verts):'\
                    .format(num_attrs,num_elems,byteStride,num_verts))

            attributeBytes = adata.tobytes()
            bv_id = writer.appendMainBufferAndGetBufferView(attributeBytes,byteStride=byteStride)
            attribute_2_accid = {}

            offset = 0
            for i in range(num_attrs):
                if len(attribute_data[i].shape) == 1 or attribute_data[i].shape[1] == 1: the_type='SCALAR'
                elif attribute_data[i].shape[1] == 2: the_type='VEC2'
                elif attribute_data[i].shape[1] == 3: the_type='VEC3'
                elif attribute_data[i].shape[1] == 4: the_type='VEC4'
                else: assert(False)
                acc = dict(bufferView=bv_id, byteOffset=offset, componentType=5126,
                        count=num_verts, type=the_type)
                if attribute_keys[i] == 'POSITION':
                    acc['min'] = attribute_data[i].min(0).tolist()
                    acc['max'] = attribute_data[i].max(0).tolist()
                acc_id = writer.putAccessor(acc)
                attribute_2_accid[attribute_keys[i]] = acc_id
                offset += attribute_data[i].shape[1] * attribute_data[i].dtype.itemsize
            print('\t\t- Attribute BufferView:', bv_id)
            print('\t\t- Attributes-to-accessors:', attribute_2_accid)

            # Make accessor for indices.
            indices = prim['indices']
            assert(indices.min() >= 0)
            indices_max = indices.max()
            if indices_max < 2**8: indices,idx_type    = indices.astype(np.uint8),  GL_UNSIGNED_BYTE
            elif indices_max < 2**16: indices,idx_type = indices.astype(np.uint16), GL_UNSIGNED_SHORT
            else: indices,idx_type                       = indices.astype(np.uint32), GL_UNSIGNED_INT
            index_bv_id = writer.appendMainBufferAndGetBufferView(indices.tobytes(), target=int(GL_ELEMENT_ARRAY_BUFFER))
            index_acc = dict(bufferView=index_bv_id, byteOffset=0, componentType=idx_type,
                            count=len(indices), type='SCALAR')
            index_acc_id = writer.putAccessor(index_acc)

            #assert(prim['material'] == 0)
            material_id = prim['material']
            mode = prim['mode']

            print('\t\t Commiting primitive!')
            out_prims.append(dict(attributes=attribute_2_accid, indices=index_acc_id,
                material=material_id,mode=mode))

        print('\t Pushing mesh {} ({} prims)'.format(self.name,len(out_prims)))
        return writer.putMesh(dict(name=self.name, primitives=out_prims))


class GltfWriter:
    def __init__(self, outPath):
        self.path = os.path.split(outPath)[0]
        self.filename = outPath

        self.images, self.nodes, self.meshes = [],[],[]
        self.accessors, self.bufferDatas, self.bufferViews = [],[],[]
        self.buffers = [] # NOTE: we always use self.mainBufferData as first!
        self.materials = [{}]
        #self.materials = []
        self.textures = []
        self.samplers = [{}]
        self.scenes = []
        self.mainBufferData = bytearray()

        if outPath.endswith('glb'):
            self.binary = True
        elif outPath.endswith('gltf'):
            self.binary = False
        else: assert(False)

        self.mainBufferIdx = 0

    # Note: In the general case primitives can share vertex data thorugh same accessors.
    # But to simplify this code, we only allow 1-1 mappings.
    # Note: Returns accessor id. Will create bufferView + accessor (and push data to main buffer)
    def putVertexData(self, data, semantic):
        pass

    def putMesh(self, mesh):
        self.meshes.append(mesh)
        return len(self.meshes)-1

    def putScene(self, sceneSpec):
        self.scenes.append(sceneSpec)
        return len(self.scenes)-1

    def putMaterial(self, mat):
        self.materials.append(mat)
        return len(self.materials)-1

    def putNode(self, nodeSpec):
        assert('mesh' in nodeSpec) # Not really ... we can have empty nodes.
        assert(nodeSpec['mesh'] <= len(self.meshes)) # Put nodes only after meshes
        self.nodes.append(nodeSpec)
        return len(self.nodes)-1

    def putImage(self, imageData, ext='jpg'):
        assert(imageData.shape[-1] == 3)
        if self.binary:
            _,img_bytes = cv2.imencode('.jpg', cv2.cvtColor(imageData,cv2.COLOR_RGB2BGR))
            bv = self.appendMainBufferAndGetBufferView(img_bytes)
            spec = dict(bufferView=bv, mimeType='image/jpeg')
            self.images.append(spec)
        else:
            # TODO Allow base64 encoding.
            imageName = 'img_{}.{}'.format(len(self.images),ext)
            imagePath = os.path.join(self.path, imageName)
            cv2.imwrite(imagePath, imageData)
            spec = dict(uri=imagePath)
            self.images.append(spec)

        return len(self.images)-1

    def putTextureFromImage(self, imageData, sampler=0):
        source = self.putImage(imageData)
        self.textures.append(dict(source=source,sampler=sampler))
        return len(self.textures)-1

    def appendMainBufferAndGetBufferView(self, data, byteStride=None, target=34962):
        # TODO Handle alignment
        self.bufferViews.append({
                'buffer': self.mainBufferIdx,
                'byteOffset': len(self.mainBufferData),
                'byteLength': len(data),
                'target': target })
        if byteStride:
            self.bufferViews[-1]['byteStride'] = byteStride
        self.mainBufferData.extend(data)
        return len(self.bufferViews)-1

    def putAccessor(self, acc):
        self.accessors.append(acc)
        return len(self.accessors)-1

    def write(self):

        assert(len(self.bufferDatas) == 0) # We only support self.mainBuffer for now

        # note: we do 'buffers' below, depending on binary or not.
        jobj = {}
        jobj['asset'] = {'version': '2.0'}
        for k in ('accessors bufferViews nodes scenes meshes textures samplers ' + \
                'images materials').split(' '):
            if hasattr(self,k): jobj[k] = getattr(self,k)

        if self.binary:
            data1 = self.mainBufferData
            while len(data1) % 4 != 0: data1 += (b'\x00')

            assert(len(self.buffers) == 0)
            #self.buffers = [{'byteLength': len(self.mainBufferData)}]
            self.buffers = [{"name":"binary_glTF", "byteLength":len(data1)}]
            jobj['buffers'] = self.buffers

            #data0 = json.dumps(jobj).replace(' ','').encode('ascii')
            data0 = json.dumps(jobj).encode('ascii')

            while len(data0) % 4 != 0: data0 += (b'\x20')

            chunk0 = bytearray()
            chunk0.extend((len(data0)).to_bytes(4,'little'))
            chunk0.extend(b'JSON')
            chunk0.extend(data0)
            chunk0 = bytes(chunk0)

            chunk1 = bytearray()
            chunk1.extend((len(data1)).to_bytes(4,'little'))
            chunk1.extend(b'BIN\0')
            chunk1.extend(data1)
            chunk1 = bytes(chunk1)

            header = bytearray()
            header.extend(b'glTF')
            header.extend((2).to_bytes(4, 'little'))
            length = 12 + len(chunk0) + len(chunk1)
            header.extend((length).to_bytes(4, 'little'))

            bits = header + chunk0 + chunk1

            with open(self.filename, 'wb') as fp:
                fp.write(bits)
            print('\n - Done writing', self.filename, '(len {} {})'.format(length,len(bits)), '\n')

        else:
            buffer_bin_name = self.filename.rsplit('.',1)[0]+'.bin'
            with open(buffer_bin_name, 'wb') as fp:
                fp.write(self.mainBufferData)

            jobj['buffers'] = [{
                'byteLength': len(self.mainBufferData),
                'uri': buffer_bin_name
            }]
            json_data = json.dumps(jobj).encode('ascii')

            with open(self.filename,'wb') as fp:
                fp.write(json_data)


def triangulate_poly(verts):
    from scipy.spatial import Delaunay
    return Delaunay(verts).simplices

# Assume verts is 2d, but inds is 1d
# Simple algo:
#    double up the verts, on the second copy lift a couple of meters.
#    for each input tri, we need four output tris:
#           1 copy of the original
#           1 for the new raised face.
#           2 for the two new extruded faces.
# TODO: it is incorrect: must use new arg 'poly' to walk along original indices.
#       need 2-faces per poly-segment, ~6 per tri face.
def extrude_triangulated_polygon(poly, verts, faces, offset=(0,0,10)):
    if len(faces.shape) == 1: faces = faces.reshape(-1, 3)
    assert(faces.shape[1] == 3)
    nv,nf = len(verts), len(faces)
    print('nf',nf,'nv',nv)
    verts = np.tile(verts, (2,1))
    faces = np.tile(faces, (4,1))

    verts[nv:] += offset

    # Old face.
    faces[nf*0:nf*1]
    # Top face.
    faces[nf*1:nf*2] += nv
    # Side face 1.
    faces[nf*2:nf*3, 2] += nv-1
    # Side face 2.
    faces[nf*3:nf*4, 1:3] += nv-1

    return verts, faces.reshape(-1)

def triangulate_poly_to_3d(verts, offset):
    from scipy.spatial import Delaunay
    nv = len(verts)
    verts = verts.repeat(2,0)
    verts[nv:] += offset

    quads = Delaunay(verts).simplices
    faces = []
    for quad in quads:
        faces.append(quad[0:3])
        faces.append(quad[1:4])
        faces.append(np.concatenate([quad[0:1],quad[2:4]]))
    return verts, np.vstack(faces)


def make_gltf_on_polys(outname='out/buildings.glb'):
    polys = get_kml_polygons(open('/home/slee/Downloads/Untitled map.kml').read())

    writer = GltfWriter(outname)

    mesh = OutputMesh('buildings')

    all_verts = []
    all_inds = []
    n_verts = 0

    #for poly in polys[0:1]:
    for poly in polys:
        verts = poly['pts'].astype(np.float32)
        poly_verts = verts[:, :2]

        faces = triangulate_poly(poly_verts)
        print('triangulated faces', faces)

        verts,indices = extrude_triangulated_polygon(poly_verts,verts,faces, offset=(0,0,.0001))
        print(indices)
        '''
        verts,indices = triangulate_poly_to_3d(verts, (0,0,0.0001))
        print('inds',indices)
        indices = indices.ravel()
        '''

        indices += n_verts

        n_verts += len(verts)
        all_verts.append(verts)
        all_inds.append(indices)

    all_verts = np.vstack(all_verts)
    all_inds = np.concatenate(all_inds)

    all_verts = all_verts - all_verts.mean(0)
    all_verts = 2 * all_verts / abs(all_verts.max())

    mesh.pushPrimitive(dict(POSITION=all_verts), indices=all_inds, mode=GL_TRIANGLES)

    mesh = mesh.finish(writer)
    node = writer.putNode(dict(mesh=mesh))
    writer.putScene(dict(nodes=[node]))

    writer.write()

'''
make_gltf_on_polys()
'''
