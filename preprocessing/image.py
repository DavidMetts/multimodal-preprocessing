"""Mask designed only for the current fixed 512 x 640 generator canvas."""
from io import BytesIO
import numpy as np
from PIL import Image

RULE = "512x640 RGB; x=189..323, y=61..387; exclude background (229,231,235), cone (199,138,69), outline (156,101,47)"

def extract(data):
    with Image.open(BytesIO(data)) as source:
        if source.format != 'PNG' or source.size != (512,640):
            raise ValueError('Expected a 512 x 640 generator PNG; arbitrary images are not supported.')
        if source.mode not in ('RGB','RGBA'):
            raise ValueError('Expected RGB PNG.')
        if source.mode=='RGBA' and np.any(np.array(source.getchannel('A')) != 255):
            raise ValueError('Transparent pixels are unsupported.')
        image=np.array(source.convert('RGB'))
    yy,xx=np.indices(image.shape[:2]); mask=(xx>=189)&(xx<=323)&(yy>=61)&(yy<=387)
    for color in ((229,231,235),(199,138,69),(156,101,47)):
        mask &= np.any(image!=color,axis=2)
    if not mask.any():
        raise ValueError('No ice-cream pixels selected by the controlled-image mask.')
    means=image[mask].mean(axis=0); area=float(mask.mean())
    overlay=(image.astype(float)*.35).astype('uint8'); overlay[mask]=image[mask]
    raw=dict(zip(('mean_red','mean_green','mean_blue'),means.tolist()))
    raw.update(selected_pixel_count=int(mask.sum()), canvas_pixel_count=int(mask.size),area_fraction=area)
    normalized={k+'_normalized':v/255 for k,v in zip(('mean_red','mean_green','mean_blue'),means.tolist())}
    normalized['area_fraction_normalized']=area
    return raw|normalized, overlay, mask
