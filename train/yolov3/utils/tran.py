import numpy as np
import math
from PIL import Image

def tran (aff_in, aff_ar):
    aff_out = np.linalg.solve(aff_ar,aff_in)
    return aff_out

def get_array(angle,shx,shy,size):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cx=(size[0]-1)/2
    cy=(size[1]-1)/2
    Rotation=np.asarray([[cos_angle,-sin_angle,cx-cos_angle*cx+sin_angle*cy],[sin_angle,cos_angle,cy-sin_angle*cx-cos_angle*cy],[0,0,1]])
    Shear1=np.asarray([[1,shx,-shx*cy],[0,1,0],[0,0,1]])
    Shear2=np.asarray([[1,0,0],[shy,1,-shy*cx],[0,0,1]])
    return np.dot(Shear1,Shear2, Rotation)

def affine(image,bbox,angle,shx,shy):
    xmin,ymin,xmax,ymax=bbox
    aff_in=np.asarray([[xmin,xmax,xmin,xmax],[ymin,ymin,ymax,ymax],[1,1,1,1]])
    size=image.size
    aff_ar=get_array(angle,shx,shy,size)
    aff_ar_2=get_array(-angle,-shx,-shy,size)
    rz_image=image.transform(image.size, Image.AFFINE, (aff_ar[0,0],aff_ar[0,1],aff_ar[0,2],aff_ar[1,0],aff_ar[1,1],aff_ar[1,2]), resample=Image.BICUBIC)
    aff_out=tran(aff_in,aff_ar)
    nx=aff_out[0]
    ny=aff_out[1]
    nxmax=max(nx)
    nymax=max(ny)
    nxmin=min(nx)
    nymin=min(ny)
    return rz_image,[nxmin,nymin,nxmax,nymax]