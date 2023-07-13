img_path = '/root/autodl-tmp/CAP_mfd_dataset/Tr00/img/img_0.jpg'
bbox = [[142,
        1924,
        175,
        39],[
        43,
        625,
        1373,
        225
      ],[
        284,
        948,
        894,
        91
      ],[
        1209,
        1095,
        107,
        29
      ],[
        1101,
        1154,
        240,
        46
      ],] #list of list, [[x,y,w,h],[x,y,w,h],...]
dst_path = '/root/autodl-tmp/simple_visualize/img_0.jpg'
# visualize the bbox on the image and save to dst_path:
import cv2
import numpy as np
import numba

def ImageRotate(image,angle,scale,tgtw,tgth):
    h, w = image.shape[:2]
    nh,nw = int(h*scale),int(w*scale)
    #image = cv2.resize(image,(nw,nh))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=scale)
    M[0, 2] += (tgtw / 2) - center[0]
    M[1, 2] += (tgth / 2) - center[1]
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(tgtw,tgth), borderValue=(255, 255, 255))
    return image_rotation

def visualize_bbox_and_save(img_path,bbox,dst_path):
    img = cv2.imread(img_path)
    # img = ImageRotate(img,1.5,1,9000,6000)
    print(np.mean(img))
    print(img.shape)
    print(type(img))
    new_img = img.copy()
    for box in bbox:
        x,y,w,h = box
        print(w)
        print(h)
        
        # dirctly draw the bbox on the image:
        for i in range(w):
            new_img[y,x+i] = [0,0,255] # BGR
            new_img[y+h,x+i] = [0,0,255]
        for i in range(h):
            new_img[y+i,x] = [0,0,255]
            new_img[y+i,x+w] = [0,0,255]
    cv2.imwrite(dst_path,new_img)
visualize_bbox_and_save(img_path,bbox,dst_path)