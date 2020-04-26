import gdal
import numpy as np
import argparse
import os, sys, time
import time
import pylas

import cv2
from matplotlib.cm import inferno

parser = argparse.ArgumentParser()
parser.add_argument('--templateTiff', required=True)
parser.add_argument('--lidarDir', required=True)
parser.add_argument('--N', default=9999999999, type=int)

gdal_datatype = gdal.GDT_Float32

def get_points_from_las(fn, N=99999999, stride=1):
    with pylas.open(fn) as fp:
        las = fp.read()
        x,y,z = las.x[:N:stride],las.y[:N:stride],las.z[:N:stride]
        points = np.stack( (x,y,z) , -1 )
    return points

# Apply the geotransform (utm -> pixel)
def normalize_points(pts, dset, xform):
    print(' - Before:\n',pts[0:10],pts[-11:])
    t = np.array( (*xform[0:2, 2], 0) , dtype=np.float32 )
    R = np.eye(3)
    R[:2,:2] = xform[:2,:2]
    #pts = (pts - t) @ np.linalg.inv(R).T
    pts = (pts - t) @ (R.T)
    print(' - After:\n',pts[0:10],pts[-11:])
    return pts


def write_tiff_with_cloud(dset, points, srcDset=None):
    dw,dh = dset.RasterXSize, dset.RasterYSize
    x1,y1,x2,y2 = *points[:,0:2].min(0), *points[:,0:2].max(0)
    x1,x2 = max(min(x1,dw),0),max(min(x2,dw),0)
    y1,y2 = max(min(y1,dh),0),max(min(y2,dh),0)
    print(x1,x2,y1,y2)
    if x1>=x2 or y1>=y2 or x2<=0 or y2<=0 or x1>=dw or y2>=dh:
        print(' - point cloud has invalid bounds w.r.t dataset:',
                x1,y1,x2,y2, 'dset size:',dw,dh)
        return False

    w,h = int(x2-x1+.99), int(y2-y1+.99)
    N = len(points)
    print(' - Making raster of size {} {} from cloud of size {}.'.format(h,w,N))

    # Move to local coords
    points = points - (x1,y1,0)
    points_quantized = points[:, 0:2].astype(int)
    zs = points[:, 2]

    doMedianFilter = True
    doMedianSampling = False
    doAverageSampling = False
    doAverageSampling = True
    #doMaxSampling = True

    if doMedianSampling:
        # We use three channels and take median in case of multiple samples for one pixel.
        arr = np.zeros( (h,w,3) , dtype=np.float32 )

        i = 0
        for ((x,y),z) in zip(points_quantized, zs):
            if arr[y,x,0] == 0:
                arr[y,x,0] = z
            else:
                if arr[y,x,1] == 0:
                    arr[y,x,1] = z
                else:
                    arr[y,x,2] = z
            if i % 10000 == 0:
                print(i,'/',N,'({}%)'.format(100*i/N))
            i += 1

        no_first = np.argwhere(arr[...,0] == 0).T
        arr[no_first[0],no_first[1]] = -1

        no_second = np.argwhere(arr[...,1] == 0).T
        arr[no_second[0],no_second[1]] = arr[no_second[0],no_second[1],0:1] # Replicate

        no_third = np.argwhere(arr[...,2] == 0).T
        arr[no_third[0],no_third[1],2] = arr[no_third[0],no_third[1],0:2].min(1) # Take lowest.

        arr = np.median(arr, 2)
        #arr = np.mean(arr, 2)
        #arr = np.max(arr, 2)

    elif doAverageSampling:
        # We mix multiple samples for one pixel.
        arr = np.zeros( (h,w) , dtype=np.float32 )
        counts = np.zeros( (h,w) , dtype=np.int32 )
        i = 0
        for ((x,y),z) in zip(points_quantized, zs):
            arr[y,x] += z
            counts[y,x] += 1
            if i % 10000 == 0: print(i,'/',N,'({}%)'.format(100*i/N))
            i += 1
        counts = np.clip(counts, a_min=1, a_max=99999)
        arr = (arr / counts.astype(np.float32)).astype(np.float32)
    else:
        assert(doMaxSampling)
        arr = np.zeros( (h,w) , dtype=np.float32 )
        i = 0
        for ((x,y),z) in zip(points_quantized, zs):
            arr[y,x] = max(arr[y,x],z)
            if i % 10000 == 0: print(i,'/',N,'({}%)'.format(100*i/N))
            i += 1

    if doMedianFilter: arr = cv2.medianBlur(arr, 3)
    #arr[arr>100] = 100

    #dimg = (inferno((arr - arr.min()) / (arr.max()-arr.min())) * 255)[...,0:3].astype(np.uint8)
    #dimg = cv2.cvtColor(dimg,cv2.COLOR_RGB2BGR)
    #cv2.imshow('dimg', dimg); cv2.waitKey(0); cv2.destroyAllWindows()

    dset.WriteRaster(int(x1),int(y1), w,h, arr.tobytes())

    return arr


creation_options = \
    'BLOCKXSIZE=256 BLOCKYSIZE=256 TILED=YES TILING=YES BIGTIFF=YES COMPRESS=LZW'.split(' ')

def main(args):
    src_dset = gdal.Open(args.templateTiff)
    driver = gdal.GetDriverByName('GTiff')
    dst_tiff_name = os.path.join(args.lidarDir, 'depth.tiff')
    dst_dset = driver.Create(dst_tiff_name,
            src_dset.RasterXSize, src_dset.RasterYSize, 1,
            gdal_datatype, creation_options)

    dst_dset.SetProjection(src_dset.GetProjection())
    dst_dset.SetGeoTransform(src_dset.GetGeoTransform())

    base_xform = dst_dset.GetGeoTransform()
    base_xform = np.array(base_xform)[ [1,2,0, 4,5,3] ].reshape(2,3)
    print(' - Base Xform:\n', base_xform)

    dband = dst_dset.GetRasterBand(1)
    dband.SetNoDataValue(-1)


    for filename in os.listdir(args.lidarDir):
        if filename.endswith('.las') or filename.endswith('.laz'):
            filename = os.path.join(args.lidarDir, filename)

            points = get_points_from_las(filename, N=args.N)

            points = normalize_points(points, dst_dset, base_xform)

            write_tiff_with_cloud(dst_dset, points)

if __name__ == '__main__':
    main(parser.parse_args())
