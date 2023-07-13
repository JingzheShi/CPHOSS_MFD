import json
import os
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import numba
from copy import deepcopy
classname_id = {'embedded': 1, 'isolated': 2}


@numba.jit(nopython=True)
def iou_max(bboxes,new_bbox):
    iou_max = 0
    for bbox in bboxes:
        # bbox = [x,y,w,h]
        # new_bbox = [x,y,w,h]
        # intersection:
        intersection_xmin = max(bbox[0],new_bbox[0])
        intersection_ymin = max(bbox[1],new_bbox[1])
        intersection_xmax = min(bbox[0]+bbox[2],new_bbox[0]+new_bbox[2])
        intersection_ymax = min(bbox[1]+bbox[3],new_bbox[1]+new_bbox[3])
        intersection_w = max(0,intersection_xmax-intersection_xmin)
        intersection_h = max(0,intersection_ymax-intersection_ymin)
        intersection_area = intersection_w*intersection_h
        # union:
        union_area = bbox[2]*bbox[3]+new_bbox[2]*new_bbox[3]-intersection_area
        # iou:
        iou = intersection_area/union_area
        if iou > iou_max:
            iou_max = iou
    return iou_max

@numba.jit(nopython=True)
def iom_max(bboxes,new_bbox):
    iom_max = 0
    for bbox in bboxes:
        intersection_xmin = max(bbox[0],new_bbox[0])
        intersection_ymin = max(bbox[1],new_bbox[1])
        intersection_xmax = min(bbox[0]+bbox[2],new_bbox[0]+new_bbox[2])
        intersection_ymax = min(bbox[1]+bbox[3],new_bbox[1]+new_bbox[3])
        intersection_w = max(0,intersection_xmax-intersection_xmin)
        intersection_h = max(0,intersection_ymax-intersection_ymin)
        intersection_area = intersection_w*intersection_h
        area = bbox[2]*bbox[3] if bbox[2]*bbox[3] < new_bbox[2]*new_bbox[3] else new_bbox[2]*new_bbox[3]
        iom = intersection_area/area
        if iom > iom_max:
            iom_max = iom
    return iom_max


class Org2Cap():
    def __init__(self,args,database):
        self.args = args
        self.org_img_path = args.img_path
        self.dst_img_path = args.dst_img_path
        self.dst_json_path = args.dst_json_path
        self.database = database.copy()
        orglenth = len(database['annotations'])
        for annotation in self.database['annotations']:
            annotation['bbox'] = np.array(annotation['bbox'])
        for idx in range(orglenth):
            idx = orglenth-1-idx
            annotation = database['annotations'][idx]
            if annotation['bbox'][2] / annotation['bbox'][3] <= self.args.min_w_h_ratio:
                self.database['annotations'].pop(idx)
            elif args.embedded_only and annotation["category_id"] == 2:
                self.database['annotations'].pop(idx)
        self.annotation_num = len(self.database['annotations'])
        print("We use {} formulas".format(self.annotation_num))
        print("Of the {} valid formulas (whose w/h >= min_w_h_ratio = {} and is of type embedded)".format(len(self.database['annotations']),self.args.min_w_h_ratio))
        print("Of the {} formulas".format(orglenth))
        self.random_order = np.random.permutation(self.annotation_num)
        # random_order: an array of random order of annotation index
        self.boxcounter = 0
        self.Delta = args.Delta
        self.go_over_img_round_counter = 0
        self.num_imgs = -1
    
    def get_and_process(self,bid):
        bbox = self.database['annotations'][bid]['bbox'].copy()
        btype = self.database['annotations'][bid]['category_id']
        # generate a random float in [-0.1,0.1]
        rotation_angle = np.random.rand()*2*self.args.left_and_right_max_rotation_angle-self.args.left_and_right_max_rotation_angle
        scale = np.random.rand()*(self.args.max_formula_resize_ratio-self.args.min_formula_resize_ratio)+self.args.min_formula_resize_ratio
        return (bbox * scale * self.args.bbox_enlarge_ratio).astype(np.int32), rotation_angle, scale, btype
    
    
    
    def random_bbox(self,imgid):
        baseImg = 255 * np.ones((self.args.img_height, self.args.img_width, 3), dtype=np.int32)
        MAXFailForOnePaper = self.args.max_try
        MAXFailForOneFormula = self.args.max_try_for_one_formula
        failcounter = 0
        known_bboxes = np.empty([0,4]).astype(int)
        known_bboxesdctlst = []
        formulacounter = 0
        target_formula_num = np.random.randint(self.args.less_formula_num,self.args.max_formula_num)
        if np.random.randn() < self.args.less_formula_possibility:
            target_formula_num = np.random.randint(0,self.args.less_formula_num)
        while((failcounter < MAXFailForOnePaper) and (formulacounter < target_formula_num)):
            bid = self.random_order[self.boxcounter]
            bboxXYWH,rot,scale,btype = self.get_and_process(bid)
            randomSampleCounter = 0
            succeed = False
            while (randomSampleCounter < self.args.max_try_for_one_formula):
                randomSampleCounter += 1
                if self.args.img_width <= bboxXYWH[2] or self.args.img_height <= bboxXYWH[3]:
                    succeed = False
                    break
                x = np.random.randint(0, self.args.img_width - bboxXYWH[2])
                y = np.random.randint(0, self.args.img_height - bboxXYWH[3])
                newbbox = np.array([x, y, bboxXYWH[2], bboxXYWH[3]]).astype(int)
                assert x>=0, 'debugging'
                assert y>=0, 'debugging'
                if (len(known_bboxes)==0) or ((iou_max(known_bboxes,newbbox) < self.args.max_iou) and (iom_max(known_bboxes,newbbox) < self.args.max_iom)):
                    known_bboxes = np.concatenate((known_bboxes,newbbox[np.newaxis,...]),axis=0)
                    known_bboxesdctlst.append(dict(
                        id = bid,
                        rot = rot,
                        scale = scale,
                        x = int(x),
                        y = int(y),
                        w = int(bboxXYWH[2]),
                        h = int(bboxXYWH[3]),
                        btype=btype,
                        ))
                    succeed = True
                    break
                else:
                    randomSampleCounter += 1
            if 0:
                if not succeed:
                    failcounter += 1
                    self.boxcounter = (self.boxcounter + 1) % len(self.random_order)
                else:
                    failcounter = 0
                    formulacounter += 1
                    self.random_order = np.delete(self.random_order,self.boxcounter)
                    if len(self.random_order) == 0:
                        self.random_order = np.random.permutation(self.annotation_num)
                        self.go_over_img_round_counter += 1
                        print("Now we have went over all the {} formulas, and start over again.".format(self.annotation_num))
                    self.boxcounter = self.boxcounter % len(self.random_order)
            if 1:
                if not succeed:
                    failcounter += 1
                    self.random_order = np.delete(self.random_order,self.boxcounter)
                    if len(self.random_order) == 0:
                        self.random_order = np.random.permutation(self.annotation_num)
                        self.go_over_img_round_counter += 1
                        print("Now we have went over all the {} formulas, and start over again.".format(self.annotation_num))
                    self.boxcounter = self.boxcounter % len(self.random_order)
                else:
                    failcounter = 0
                    formulacounter += 1
                    self.random_order = np.delete(self.random_order,self.boxcounter)
                    if len(self.random_order) == 0:
                        self.random_order = np.random.permutation(self.annotation_num)
                        self.go_over_img_round_counter += 1
                        print("Now we have went over all the {} formulas, and start over again.".format(self.annotation_num))
                    self.boxcounter = self.boxcounter % len(self.random_order)
        self.boxcounter = 0
        return known_bboxesdctlst,formulacounter
            
    def regular_bbox(self,imgid):
        baseImg = 255 * np.ones((self.args.img_height, self.args.img_width, 3), dtype=np.int32)
        MAXFailForOnePaper = self.args.max_try
        MAXFailForOneFormula = self.args.max_try_for_one_formula
        failcounter = 0
        known_bboxes = np.empty([0,4]).astype(int)
        known_bboxesdctlst = []
        formulacounter = 0
        target_formula_num = np.random.randint(self.args.less_formula_num,self.args.max_formula_num)
        # if np.random.randn() < self.args.less_formula_possibility:
        # TODO we always use less formula when dealing with regular_bboxes.
        target_formula_num = np.random.randint(10,self.args.less_formula_num)
        old_x = np.random.randint(0,self.args.img_width)
        old_y = np.random.randint(0,self.args.img_height)
        old_width = 0
        old_height = 0
        while((failcounter < MAXFailForOnePaper) and (formulacounter < target_formula_num)):
            bid = self.random_order[self.boxcounter]
            bboxXYWH,rot,scale,btype = self.get_and_process(bid)
            randomSampleCounter = 0
            succeed = False
            
            while (randomSampleCounter < self.args.max_try_for_one_formula):
                randomSampleCounter += 1
                if self.args.img_width <= bboxXYWH[2] or self.args.img_height <= bboxXYWH[3]:
                    succeed = False
                    break
                if (self.args.img_width - old_x - old_width <= bboxXYWH[2]) and (self.args.img_height - old_y - old_height <= bboxXYWH[3]):
                    succeed = False
                    break
                elif (self.args.img_width - old_x - old_width <= bboxXYWH[2]) and (self.args.img_height - old_y - old_height > bboxXYWH[3]):
                    old_x = np.random.randint(0,self.args.img_width)
                    continue
                elif (self.args.img_width - old_x - old_width > bboxXYWH[2]) and (self.args.img_height - old_y - old_height <= bboxXYWH[3]):
                    old_y = np.random.randint(0,self.args.img_height)
                    continue
                else:
                    x = np.random.randint(old_x + old_width - self.Delta, old_x + old_width + self.Delta)
                    y = np.random.randint(old_y + old_height - self.Delta, old_y + old_height + self.Delta)
                x = 0 if x < 0 else x
                x = self.args.img_width - bboxXYWH[2] if x >= self.args.img_width - bboxXYWH[2] else x
                y = 0 if y < 0 else y
                y = self.args.img_height - bboxXYWH[3] if y >= self.args.img_height - bboxXYWH[3] else y
                newbbox = np.array([x, y, bboxXYWH[2], bboxXYWH[3]]).astype(int)
                assert x>=0, 'debugging'
                assert y>=0, 'debugging'
                if (len(known_bboxes)==0) or ((iou_max(known_bboxes,newbbox) < self.args.max_iou) and (iom_max(known_bboxes,newbbox) < self.args.max_iom)):
                    known_bboxes = np.concatenate((known_bboxes,newbbox[np.newaxis,...]),axis=0)
                    known_bboxesdctlst.append(dict(
                        id = bid,
                        rot = rot,
                        scale = scale,
                        x = int(x),
                        y = int(y),
                        w = int(bboxXYWH[2]),
                        h = int(bboxXYWH[3]),
                        btype = btype,
                        ))
                    succeed = True
                    break
                else:
                    randomSampleCounter += 1
            if 0:
                if not succeed:
                    failcounter += 1
                    self.boxcounter = (self.boxcounter + 1) % len(self.random_order)
                else:
                    failcounter = 0
                    formulacounter += 1
                    self.random_order = np.delete(self.random_order,self.boxcounter)
                    if len(self.random_order) == 0:
                        self.random_order = np.random.permutation(self.annotation_num)
                        self.go_over_img_round_counter += 1
                        print("Now we have went over all the {} formulas, and start over again.".format(self.annotation_num))
                    self.boxcounter = self.boxcounter % len(self.random_order)
            if 1:
                if not succeed:
                    failcounter += 1
                    self.random_order = np.delete(self.random_order,self.boxcounter)
                    if len(self.random_order) == 0:
                        self.random_order = np.random.permutation(self.annotation_num)
                        self.go_over_img_round_counter += 1
                        print("Now we have went over all the {} formulas, and start over again.".format(self.annotation_num))
                    self.boxcounter = self.boxcounter % len(self.random_order)
                else:
                    old_x = x
                    old_y = y
                    failcounter = 0
                    formulacounter += 1
                    self.random_order = np.delete(self.random_order,self.boxcounter)
                    if len(self.random_order) == 0:
                        self.random_order = np.random.permutation(self.annotation_num)
                        self.go_over_img_round_counter += 1
                        print("Now we have went over all the {} formulas, and start over again.".format(self.annotation_num))
                    self.boxcounter = self.boxcounter % len(self.random_order)
        self.boxcounter = 0
        return known_bboxesdctlst,formulacounter    
    
    def generate_and_save_img_and_annotation_according_to_bboxes_dct_lst_lst(self,bboxes_dct_lst_lst):
        json_annotation = dict(
            info = 'COCO from created',
            liscense = 'MIT',
            images = [],
            annotations = [],
            categories = [{'id':1,'name':'embedded'},
                          {'id':2,'name':'isolated'}]
        )
        bbox_idx = 0
        print("Now generating and saving the images and annotations according to previous information.")
        for image_idx in tqdm(range(self.num_imgs)):
            img_filename = 'img_{}.jpg'.format(image_idx)
            img = 255 * np.ones((self.args.img_height, self.args.img_width, 3), dtype=np.int32)
            json_annotation['images'].append(dict(
                height = self.args.img_height,
                width = self.args.img_width,
                id = image_idx,
                filename = img_filename,
            ))
            for bbox_dct in bboxes_dct_lst_lst[image_idx]:
                x = bbox_dct['x']
                y = bbox_dct['y']
                w = bbox_dct['w']
                h = bbox_dct['h']
                annotation = dict(
                    id = bbox_idx,
                    category_id = bbox_dct['btype'],
                    image_id = image_idx,
                    segmentation = [x,y,x+w+1,y,x+w+1,y+h+1,x,y+h+1],
                    bbox = [x,y,w,h],
                    iscrowd = 0,
                    area = float(w*h),
                )
                json_annotation['annotations'].append(annotation)
                org_bid = bbox_dct['id']
                rot = bbox_dct['rot']
                scale = bbox_dct['scale']
                org_imgid = self.database['annotations'][org_bid]['image_id']
                img[y:y+h,x:x+w,:] = self.cut_and_rotate_and_scale(org_imgid,org_bid,rot,scale,w,h).copy()
                bbox_idx += 1
            cv2.imwrite(os.path.join(self.args.dst_img_path,img_filename),img)
        
        json.dump(json_annotation,open(self.args.dst_json_path,'w'),ensure_ascii=False,indent=2)
        print("Done.")
        
                
                
            
    def cut_and_rotate_and_scale(self,org_imgid,org_bid,rot,scale,tgtw,tgth):
        img = cv2.imread(os.path.join(self.args.img_path,self.database['images'][org_imgid]['file_name']))
        image = img[self.database['annotations'][org_bid]['bbox'][1]:self.database['annotations'][org_bid]['bbox'][1]+self.database['annotations'][org_bid]['bbox'][3],
                            self.database['annotations'][org_bid]['bbox'][0]:self.database['annotations'][org_bid]['bbox'][0]+self.database['annotations'][org_bid]['bbox'][2],:]
        cv2.imwrite('/root/autodl-tmp/debugging.jpg',image)
       # assert False,'debugging'        
        
        # rotate, resize, than pad with (255,255,255).
        h, w = image.shape[:2]
        #image = cv2.resize(image,(nw,nh))
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center=center, angle=-rot, scale=scale)
        M[0, 2] += (tgtw / 2) - center[0]
        M[1, 2] += (tgth / 2) - center[1]
        image_rotation = cv2.warpAffine(src=image, M=M, dsize=(tgtw,tgth), borderValue=(255, 255, 255))
        return image_rotation
        
    def generate_and_save(self):
        print("Now start generating informations.")
        
        bboxes_dct_lst_lst = []
        formulacounterlst=[]
        generated_image_id = -1
        while self.go_over_img_round_counter < self.args.how_many_times_through_capdb:
            generated_image_id += 1
        # for generated_image_id in tqdm(range(self.args.num_imgs)):
            #generate a random number in [0,1], if less than 0.2, then use random_bbox, else use random_formula
            random_number = np.random.rand()
            new_bboxes_dct_lst,formulacounter = self.random_bbox(generated_image_id) if random_number < self.args.RandomRatio else self.regular_bbox(generated_image_id)
            bboxes_dct_lst_lst.append(new_bboxes_dct_lst)
            formulacounterlst.append(formulacounter)
        print("Finished generating informations.")
        print("Altogether there are {} images.".format(generated_image_id + 1))
        self.num_imgs = generated_image_id + 1
        self.generate_and_save_img_and_annotation_according_to_bboxes_dct_lst_lst(bboxes_dct_lst_lst)
        
        return formulacounterlst
    
        





if __name__ == "__main__":
    parser = ArgumentParser()
    dirname = "Va01"
    parser.add_argument('--img_path', type=str, default='/root/autodl-tmp/nature21/mfd_dataset/'+dirname+'/img/', help='Your split image path')
    parser.add_argument('--org_json_path', type=str, default='/root/autodl-tmp/nature21/mfd_dataset/'+dirname+'/train_coco.json', help='Your original json path')
    parser.add_argument('--dst_img_path', type=str, default='/root/autodl-tmp/CAP_mfd_dataset_embeddedOnly/'+dirname+'/img/', help='Your split image path')
    parser.add_argument('--dst_json_path', type=str, default='/root/autodl-tmp/CAP_mfd_dataset_embeddedOnly/'+dirname+'/train_coco.json', help='coco format json destination folder')
    # parser.add_argument('--num_imgs',type=int,default = 300, help = 'number of images to be generated')
                                    #10 for debugging.
    
    
    
    
    
    parser.add_argument('--max_try',type=int,default=150,help='maxinum number of try before giving up generating an image')
    parser.add_argument('--max_try_for_one_formula',type=int,default=20,help='maxinum number of try before giving up generating a formula')
    parser.add_argument('--max_iou',type=float,default = 0.03,help='maximum iou between two formulas')
    parser.add_argument('--max_iom',type=float,default = 0.1, help = 'maximum iom between two formulas')
    parser.add_argument('--RandomRatio',type=float,default = 0.85,help='ratio of randomly positioned formulas')
                                                #debugging
                                                
    parser.add_argument('--min_w_h_ratio',type=float,default = 3)
    parser.add_argument('--Delta',type = int, default = 150)
                                                
    parser.add_argument('--bbox_enlarge_ratio',type=float,default = 1.1,help='ratio of enlarging the bbox')
    parser.add_argument('--img_height',type=int,default = 2048)
    parser.add_argument('--img_width',type=int,default = 1447)
    
    
    parser.add_argument('--less_formula_possibility',type=float,default = 0.13,help='possibility of generating less formulas in an image')
    parser.add_argument('--less_formula_num',type=int,default=35,help='minimum number of formulas in an image')
    parser.add_argument('--max_formula_num',type=int,default=50,help='maximum number of formulas in an image')
    parser.add_argument('--min_formula_resize_ratio',type=float,default = 1.3,help='minimum resize ratio of a formula')
    parser.add_argument('--max_formula_resize_ratio',type=float,default = 2.2,help='maximum resize ratio of a formula')
    parser.add_argument('--left_and_right_max_rotation_angle',type=int,default = 2.2)
    parser.add_argument('--embedded_only',type = bool, default = True)
    parser.add_argument('--how_many_times_through_capdb',type=int,default = 2)
    
    args = parser.parse_args()
    
    
    
    
    
    img_path = args.img_path
    org_json_path = args.org_json_path
    dst_img_path = args.dst_img_path
    dst_json_path = args.dst_json_path
    
    
    database = json.load(open(org_json_path, 'r'), encoding='utf-8')
        # database: dict. keys:
        
        # info, database['info'] = 'COCO form created'
        
        # liscense, value = MIT
        
        # images, list of dict. keys: height, width, id, file_name.
            # for example, filename is 0205312-page05.jpg.
            
        # annotations, list of dict. keys: id, image_id, category_id, bbox, area, iscrowd.
            # id: bbox id
            # category_id: 1 or 2
            # image_id: image id
            # segmentation: [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]
            # bbox: [x_min, y_min, width, height]; width = x_max-x_min +1, height = y_max-y_min +1
            # iscrowd: 0
            # area: width * height

        # categories: list of dict. keys: id, name.
            # id: 1 or 2
            # name: embedded or isolated
    
    converter = Org2Cap(args,database)
    formulanumlst = converter.generate_and_save()
    print(formulanumlst)
    