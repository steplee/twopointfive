import pylas
import numpy as np
import ctypes, cv2

def get_points(lidarFile, num=-1, stride=1):
    with pylas.open(lidarFile) as fh:
        print('Points from Header:', fh.header.point_count)
        las = fh.read()
        print(las)
        print('Points from data:', len(las.points))
        print(las.points_data.point_format.dimension_names)

        N,s = num, stride
        x,y,z = las.x[0:N:s],las.y[:N:s],las.z[:N:s]
        n = len(x)

        color = None
        '''
        dset = None
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
        '''

        pts = np.stack((x,y,z), -1).astype(np.float32)
        if False:
            pts = pts - pts.mean(0)
            pts = pts / np.quantile(abs(pts), .96)
            pts = pts - pts.mean(0)
            pts = pts * 50000
            pts = pts - pts.mean(0)
            pts = pts - pts.mean(0)
        print('max/min',pts.max(0), pts.min(0))

        return pts, color
