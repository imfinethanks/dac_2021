import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import xml.dom.minidom
from PIL import ImageDraw
import torchvision.transforms as transforms
from tqdm import tqdm
import utils.tran as tran

def analyze_xml(file_path):
    meta = xml.etree.ElementTree.parse(file_path).getroot()
    size = meta.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    obj = meta.find('object')
    box = obj.find('bndbox')
    xmin = int(box.find('xmin').text)
    ymin = int(box.find('ymin').text)
    xmax = int(box.find('xmax').text)
    ymax = int(box.find('ymax').text)

    x = (xmin + xmax) / 2 / img_width
    y = (ymin + ymax) / 2 / img_height
    bb_width = (xmax - xmin) / img_width
    bb_height = (ymax - ymin) / img_height

    return np.array([[0, x, y, bb_width, bb_height]], dtype='float32')
class customDataset(Dataset):
    def __init__(self, root='files_test_txt.txt', shape=None, transform=None,is_image=True,ag=True):
        file_handle=open(root,mode='r')
        self.files=[]
        for file_names in file_handle.readlines():
            self.files.append(file_names[:-1])
        n = len(self.files)  
        self.labels = [None] * n  
        pbar = tqdm(self.files, desc='Caching labels')
        for i, file in enumerate(pbar):
            self.labels[i]=analyze_xml(file+'.xml')
        self.nSamples = len(self.files)
        self.transform = transform
        self.shape = shape
        self.is_image = is_image
        self.ag = ag
        self.flip=True
        self.scale=1
        self.scale_rate=np.log2(2) #2的几次方
        self.whc=0 #长宽比变化
        self.whc_rate=np.log2(1.5) #2的几次方
        self.move=True #不加可能导致物体在图外
        self.Noise=False
        self.Noise_rate=0
        self.Affine=0
        self.shx=1
        self.shy=1
        self.angle=np.pi/4

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if(self.is_image):
    
            targetpath = self.files[index]+'.xml'
            annotation = xml.dom.minidom.parse(targetpath).documentElement
            obj=annotation.getElementsByTagName("object")[0]
            size=annotation.getElementsByTagName("size")[0]
            width=int(size.getElementsByTagName("width")[0].childNodes[0].data)
            height=int(size.getElementsByTagName("height")[0].childNodes[0].data)
            xmin=float(obj.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin=float(obj.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax=float(obj.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax=float(obj.getElementsByTagName("ymax")[0].childNodes[0].data)
            imgpath = self.files[index]+'.jpg'
            img = Image.open(imgpath).convert('RGB')
            raw_width=width
            raw_height=height
            if self.ag:
                if self.flip:
                    if random.random()>0.5:
                        img= img.transpose(Image.FLIP_LEFT_RIGHT)
                        xmin,xmax=width-xmax,width-xmin
                whc_rate_now=1
                if self.whc>random.random():
                    whc_rate_now=2**random.uniform(-self.whc_rate,self.whc_rate)


                if self.scale>random.random():
                    max_rate=min(2**self.scale_rate,width/whc_rate_now/(xmax-xmin+1),height*whc_rate_now/(ymax-ymin+1))
                    
                    scale_rate_now=2**(random.uniform(-self.scale_rate,np.log2(max_rate)))
                    # 下面代码在max_rate<1时会出错
                    # if random.random()>0.5:
                    #     scale_rate_now=random.uniform(1,max_rate)
                    # else:
                    #     scale_rate_now=1/random.uniform(1,self.scale_rate)
                    xmin,ymin,xmax,ymax=xmin*scale_rate_now*whc_rate_now,ymin*scale_rate_now/whc_rate_now,xmax*scale_rate_now*whc_rate_now,ymax*scale_rate_now/whc_rate_now
                    width,height=int(width*scale_rate_now*whc_rate_now), int(height*scale_rate_now/whc_rate_now)
                    img = img.resize((width,height))
                p = Image.new('RGB', (width+raw_width*2, height+raw_height*2), (0, 0, 0))
                p.paste(img, (raw_width, raw_height, width+raw_width, height+raw_height))
                img=p
                if self.Affine>random.random():
                    image_raw=img.copy()
                    xmin_r,ymin_r,xmax_r,ymax_r=xmin,ymin,xmax,ymax
                    now_shx=random.uniform(-self.shx,self.shx)
                    now_shy=random.uniform(-self.shy,self.shy)
                    now_angle=random.uniform(-self.angle,self.angle)
                    img,(xmin,ymin,xmax,ymax)=tran.affine(img,[xmin+raw_width,ymin+raw_height,xmax+raw_width,ymax+raw_height],now_angle,now_shx,now_shy)
                    # b=ImageDraw.ImageDraw(img)
                    # b.rectangle((xmin,ymin,xmax,ymax),outline ='blue',width =1)
                    # b.rectangle((width/2+raw_width-10,height/2+raw_height-10,width/2+raw_width+10,height/2+raw_height+10),outline ='red',width =2)
                    # img.save("6.jpg")
                    xmin,ymin,xmax,ymax=xmin-raw_width,ymin-raw_height,xmax-raw_width,ymax-raw_height
                    if(xmax-xmin>=raw_width or ymax-ymin>=raw_height):
                        # print("=================================")
                        # print(xmin,ymin,xmax,ymax)
                        img=image_raw
                        xmin,ymin,xmax,ymax=xmin_r,ymin_r,xmax_r,ymax_r
                        # print(xmin,ymin,xmax,ymax)
                    # print(xmin,ymin,xmax,ymax)
                # print(xmin,ymin,xmax,ymax)
                # print(width,height)
                # print(raw_width,raw_height)
                if self.move:
                    px_max=int(xmin)
                    px_min=int(xmax-raw_width)
                    py_max=int(ymin)
                    py_min=int(ymax-raw_height)
                    # print(px_min,px_max)
                    # print(py_min,py_max)
                    px=random.randint(px_min,px_max)
                    py=random.randint(py_min,py_max)
                else:
                    px=(width-raw_width)/2
                    py=(height-raw_height)/2
                # print(px,py)
                img = img.crop((raw_width+px,raw_height+py,raw_width+raw_width+px, raw_height+raw_height+py)) 
                xmin-=px
                xmax-=px
                ymin-=py
                ymax-=py
                # print(xmin,ymin,xmax,ymax)
                width=raw_width
                height=raw_height
            #     x_st=random.randint(0,int(xmin))
            #     y_st=random.randint(0,int(ymin))
            #     x_ed=random.randint(int(xmax),int(width))
            #     y_ed=random.randint(int(ymax),int(height))
            #     xmin-=x_st
            #     ymin-=y_st
            #     xmax-=x_st
            #     ymax-=y_st
            #     width=x_ed-x_st
            #     height=y_ed-y_st
            #     img = img.crop((x_st,y_st,x_ed,y_ed)) 
                if self.Noise:
                    img = GaussianNoise(img,amplitude=random.uniform(0,self.Noise_rate))
            x=(xmin+xmax)/2.0/width
            y=(ymin+ymax)/2.0/height
            w=(xmax-xmin)/width
            h=(ymax-ymin)/height
            # if x>1 or y>1 or w>1 or h>1:
            #     print(x,y,w,h)

            if self.shape:
                img = img.resize(self.shape)
            
            # print(xmin,ymin,xmax,ymax)
            # print(width,height)
            # print(raw_width,raw_height)
            # a=ImageDraw.ImageDraw(img)
            # a.rectangle((xmin/width*320,ymin/height*160,xmax/width*320,ymax/height*160),outline ='black',width =1)
            # img.save("2.jpg")

            if self.transform is not None:
                img = self.transform(img)
                
            return img,torch.Tensor([[0,0,x,y,w,h]]),imgpath,((360, 640), ((0.4444444444444444, 0.5), (0.0, 0.0)))
        else:
            targetpath = self.files[index]+'.xml'
            annotation = xml.dom.minidom.parse(targetpath).documentElement
            obj=annotation.getElementsByTagName("object")[0]
            size=annotation.getElementsByTagName("size")[0]
            width=float(size.getElementsByTagName("width")[0].childNodes[0].data)
            height=float(size.getElementsByTagName("height")[0].childNodes[0].data)
            xmin=float(obj.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin=float(obj.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax=float(obj.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax=float(obj.getElementsByTagName("ymax")[0].childNodes[0].data)
            x=(xmin+xmax)/2.0/width
            y=(ymin+ymax)/2.0/height
            w=(xmax-xmin)/width
            h=(ymax-ymin)/height
            return [0,x,y,w,h]
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
if __name__ == '__main__':
    transform_train = transforms.Compose([
    transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    gg = customDataset('files_test_txt.txt',shape=(320, 160),transform=transform_train,ag=False)
    # minx=100
    # jj=0
    # for img,hh in gg:
    #     if(hh[3]>0.5 or hh[4]>0.5):
    #     # mx=hh[1]-hh[3]/2
    #         print(jj)
    #     jj+=1
    #     # if(mx<minx):
    #     #     minx=mx
    #     #     print(minx)
            

    print(gg[0])
    # print(gg[8020])
    # print(gg[15544])

# gg=customDataset('/home/fkq/data_training/')
# image,tg,path=gg[10]
# a=ImageDraw.ImageDraw(image)
# print(int(tg[0]*640-tg[2]*320),int(tg[1]*360-tg[3]*180),int(tg[0]*640+tg[2]*320),int(tg[1]*360+tg[3]*180))
# a.rectangle((int(tg[0]*640-tg[2]*320),int(tg[1]*360-tg[3]*180),int(tg[0]*640+tg[2]*320),int(tg[1]*360+tg[3]*180)),outline ='red',width =2)
# image.save('result.jpg')