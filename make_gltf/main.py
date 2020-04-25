import torch
import sys
from torch.autograd import Variable
import numpy as np
import cv2

from MegaDepth.options.train_options import TrainOptions
from MegaDepth.models.models import create_model

from .gltf_output import *


model = None

def get_depth(model, img):
    model.switch_to_eval()

    input_height = 384
    input_height = 512
    input_width  = 512
    img = cv2.resize(img, (input_width, input_height))
    input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float() / 255.
    input_img = input_img.unsqueeze(0)

    input_images = Variable(input_img.cuda() )
    pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    from matplotlib.cm import inferno
    colored = (inferno(pred_inv_depth) * 255)[...,0:3].astype(np.uint8)

    #io.imsave('demo.png', colored)
    # print(pred_inv_depth.shape)
    cv2.imshow('depth', cv2.cvtColor(colored,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img, pred_depth.detach().cpu().numpy()


if __name__ == '__main__':
    global model
    img_path = sys.argv[1]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    opt = TrainOptions().parse([])  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)
    model.netG.load_state_dict(torch.load('./saves/best_generalization_net_G.pth'))

    img,depth = get_depth(model, img)

    depth = depth.clip(min=-4,max=4)

    # Create geometry.
    depth_scale = 1
    w,h = img.shape[1], img.shape[0]
    div = 4
    w,h = w // div, h // div
    depth = depth[::div,::div]
    aspect_wh = w/h
    aspect_hw = h/w
    #yx = np.meshgrid( np.linspace(-1,1, h), np.linspace(-aspect_wh,aspect_wh, w) )
    #yx = np.meshgrid( np.linspace(-aspect_wh,aspect_wh, w), np.linspace(-1,1,h))
    yx = np.meshgrid( np.linspace(-1,1, w), np.linspace(-aspect_hw,aspect_hw,h))
    uvs = (np.stack( yx , -1 ).astype(np.float32) + 1) / 2
    pts = np.stack( (*yx, -depth*depth_scale) , -1 ).astype(np.float32)

    #uvs = uvs[:,::-1]
    #pts = pts[:,::-1]
    #pts[..., 2] *= -1
    pts[..., 1] *= -1


    ww = pts.shape[1]
    pts = pts.reshape(-1, 3)
    uvs = uvs.reshape(-1, 2)

    # Create indices for a grid.
    '''
    a-b-c
    |/|/|
    d-e-f
    |/|/|
    g-h-i

    tris = abd, bed, bce, cfe, ...

    Easiest way seems to be to loop per quad and add both tris.
    '''
    inds = []
    for y in np.arange(0,h-1):
        for x in np.arange(0,w-1):
            # ijkl in clockwise order.
            i = y*ww + x
            j = y*ww + x+1
            k = (y+1)*ww + x+1
            l = (y+1)*ww + x
            #inds.append( (i,j,l) )
            #inds.append( (j,k,l) )
            inds.append( (j,i,l) )
            inds.append( (k,j,l) )
    inds = np.array(inds, dtype=np.int32)
    inds = inds.reshape(-1)


    print(' - wh',w,h)
    print(' - inds',inds.shape, inds.min(), inds.max())
    print(' - pts',pts.shape, pts.min(0),pts.max(0))
    print(' - uvs',uvs.shape)

    # Export gltf.
    writer = GltfWriter('test.glb')
    mesh = OutputMesh('image')

    #diffuse_id = writer.putImage(img)
    tex_id = writer.putTextureFromImage(img)
    mat_id = writer.putMaterial({ 'pbrMetallicRoughness': {'baseColorTexture': {'index': tex_id} } })

    mesh.pushPrimitive(dict(POSITION=pts,TEXCOORD_0=uvs), indices=inds, mode=GL_TRIANGLES, material=mat_id)

    mesh = mesh.finish(writer)
    node = writer.putNode(dict(mesh=mesh))
    writer.putScene(dict(nodes=[node]))
    writer.write()

 
