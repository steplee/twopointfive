import numpy as np, torch
import cv2
import tview

from .lidar_stuff import get_points
from .mesher import Mesher, compute_normals

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lidarFile', default='data/USGS_LPC_MD_VA_Sandy_NCR_2014_18SUJ321308_LAS_2015.laz')
parser.add_argument('--tiff', default='/home/slee/stuff/terrapixel/terraslam/data/dc/dc.tiff')
parser.add_argument('--N', default=90000000, type=int)
parser.add_argument('--stride', default=1, type=int)
args = parser.parse_args()

#pts = get_points(args.lidarFile, args.N, args.stride)
xy = (torch.rand(10000, 2) - .5)*2 * 4
z = torch.exp(xy.norm(dim=1,keepdim=True))
z.add_(torch.randn_like(z)/10)
pts = torch.cat((xy,z),1)
nls = torch.zeros_like(pts); nls[:, -1] = 1

compute_normals(pts)

meta = {}
mesher = Mesher(meta)
mesher.run(pts,nls)
