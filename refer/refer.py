"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import sys
import os
import os.path as osp
import json
import pickle as pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np 
from pycocotools import mask


class REFER:

    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        if dataset == 'refcocog':
            print('Split by {}!'.format(splitBy))
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        
        # Robust DATA_DIR resolution
        ref_filename = 'refs(' + splitBy + ').p'
        possible_data_dirs = [
            osp.join(data_root, 'Annotations'),
            osp.join(data_root, 'annotations'),
            osp.join(data_root, dataset, 'Annotations'),
            osp.join(data_root, dataset, 'annotations'),
            osp.join(data_root, dataset),                 # /data_root/rrsis_hr
            osp.join(data_root, 'ris_lad'),
            osp.join(data_root, 'rislad'),
            osp.join(data_root, 'ris-lad'),
            osp.join(data_root, 'rrsis_hr'),
            osp.join(data_root, 'rrsis-hr'),
            osp.join(data_root, 'rrsishr'),
            osp.join(data_root, 'rrsis_d'),
            osp.join(data_root, 'rrsis-d'),
            osp.join(data_root, 'rrsisd'),
            osp.join(data_root, 'data', dataset),         # /data_root/data/rrsis_hr
            osp.join(data_root, 'data', 'rrsisd'),        # Nested data directory structure
            osp.join(data_root, 'rrsisd'),
            osp.join(data_root, 'data'),                  # /data_root/data
            data_root                                     # /data_root
        ]
        
        self.DATA_DIR = possible_data_dirs[0] # Default
        for p in possible_data_dirs:
            if osp.exists(osp.join(p, ref_filename)):
                self.DATA_DIR = p
                break

        def select_best_image_dir(candidates):
            img_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
            # 1. First priority: a directory that directly contains image files
            for p in candidates:
                if osp.exists(p) and osp.isdir(p):
                    try:
                        entries = os.listdir(p)
                        if any(e.lower().endswith(img_exts) for e in entries):
                            return p
                    except Exception:
                        pass
            # 2. Second priority: a directory containing a subdirectory with image files (e.g. images/images, images/JPEGImages)
            for p in candidates:
                if osp.exists(p) and osp.isdir(p):
                    try:
                        for sub in os.listdir(p):
                            sub_p = osp.join(p, sub)
                            if osp.isdir(sub_p):
                                sub_entries = os.listdir(sub_p)
                                if any(e.lower().endswith(img_exts) for e in sub_entries):
                                    return sub_p
                    except Exception:
                        pass
            # 3. Third priority: first non-empty directory
            for p in candidates:
                if osp.exists(p) and osp.isdir(p):
                    try:
                        if len(os.listdir(p)) > 0:
                            return p
                    except Exception:
                        pass
            return candidates[0]

        dataset_key = dataset.lower()
        if dataset_key in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
        elif dataset_key == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        elif dataset_key in ['rrsisd', 'rrsis_d', 'rrsis-d']:
            possible_image_dirs = [
                osp.join(data_root, 'images/rrsisd/JPEGImages'),
                osp.join(data_root, 'images/rrsis_d/JPEGImages'),
                osp.join(data_root, 'JPEGImages'),
                osp.join(data_root, 'images'),
                osp.join(data_root, 'Images'),
                osp.join(data_root, 'data', 'images'),
                osp.abspath(osp.join(data_root, '../JPEGImages')),
                osp.abspath(osp.join(data_root, '../images')),
            ]
            self.IMAGE_DIR = select_best_image_dir(possible_image_dirs)
        elif dataset_key in ['rrsis_hr', 'rrsis-hr', 'rrsishr']:
            possible_image_dirs = [
                osp.join(data_root, 'Images'),
                osp.join(data_root, 'images'),
                osp.join(data_root, 'data', 'images'),
                osp.join(data_root, 'images', 'rrsis_hr'),
                osp.join(data_root, 'images', 'rrsis-hr'),
                osp.abspath(osp.join(data_root, '../Images')),
                osp.abspath(osp.join(data_root, '../images')),
            ]
            self.IMAGE_DIR = select_best_image_dir(possible_image_dirs)
        elif dataset_key in ['rislad', 'ris_lad', 'ris-lad']:
            possible_image_dirs = [
                osp.join(data_root, 'images'),
                osp.join(data_root, 'Images'),
                osp.join(data_root, 'JPEGImages'),
                osp.join(data_root, 'images', 'rislad'),
                osp.join(data_root, 'images', 'ris_lad'),
                osp.join(data_root, 'data', 'images'),
                osp.abspath(osp.join(data_root, '../images')),
                osp.abspath(osp.join(data_root, '../Images')),
                osp.abspath(osp.join(data_root, '../JPEGImages')),
            ]
            self.IMAGE_DIR = select_best_image_dir(possible_image_dirs)
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        with open(ref_file, 'rb') as f:
            self.data['refs'] = pickle.load(f)

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif isinstance(ann_ids, (int, str)):
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            seg = ann.get('segmentation', None)
            if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                polygons = []
                color = []
                c = 'none'
                # polygon used for refcoco*
                for s in seg:
                    poly = np.array(s).reshape((len(s) // 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for RLE / refclef
                m = self.getMask(ref)['mask']
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        h, w = image['height'], image['width']
        seg = ann.get('segmentation', None)

        if seg is None or len(seg) == 0:
            m = np.zeros((h, w), dtype=np.uint8)
            return {'mask': m, 'area': 0.0}

        try:
            if isinstance(seg, str):
                # Single compressed RLE string
                rle = {'size': [h, w], 'counts': seg.encode('utf-8')}
                m = mask.decode(rle)
                if m.ndim == 3:
                    m = np.sum(m, axis=2)
            elif isinstance(seg, list):
                if len(seg) == 0:
                    m = np.zeros((h, w), dtype=np.uint8)
                elif isinstance(seg[0], str):
                    # List of compressed RLE strings
                    m = np.zeros((h, w), dtype=np.uint8)
                    for s in seg:
                        rle = {'size': [h, w], 'counts': s.encode('utf-8') if isinstance(s, str) else s}
                        part_m = mask.decode(rle)
                        if part_m.ndim == 3:
                            part_m = np.sum(part_m, axis=2)
                        part_m = (part_m > 0).astype(np.uint8)
                        if part_m.shape != (h, w):
                            aligned = np.zeros((h, w), dtype=np.uint8)
                            h_lim, w_lim = min(part_m.shape[0], h), min(part_m.shape[1], w)
                            aligned[:h_lim, :w_lim] = part_m[:h_lim, :w_lim]
                            part_m = aligned
                        m = np.maximum(m, part_m)
                elif isinstance(seg[0], dict):
                    # List of RLE dicts
                    m = np.zeros((h, w), dtype=np.uint8)
                    for d in seg:
                        if isinstance(d.get('counts'), list):
                            rle = mask.frPyObjects(d, h, w)
                        else:
                            counts = d.get('counts', '')
                            rle = {'size': d.get('size', [h, w]), 'counts': counts.encode('utf-8') if isinstance(counts, str) else counts}
                        part_m = mask.decode(rle)
                        if part_m.ndim == 3:
                            part_m = np.sum(part_m, axis=2)
                        part_m = (part_m > 0).astype(np.uint8)
                        if part_m.shape != (h, w):
                            aligned = np.zeros((h, w), dtype=np.uint8)
                            h_lim, w_lim = min(part_m.shape[0], h), min(part_m.shape[1], w)
                            aligned[:h_lim, :w_lim] = part_m[:h_lim, :w_lim]
                            part_m = aligned
                        m = np.maximum(m, part_m)
                else:
                    # Polygon format: [[x1, y1, x2, ...]] or [x1, y1, x2, ...]
                    if isinstance(seg[0], (int, float)):
                        seg = [seg]
                    rle = mask.frPyObjects(seg, h, w)
                    m = mask.decode(rle)
                    if m.ndim == 3:
                        m = np.sum(m, axis=2)
            elif isinstance(seg, dict):
                # RLE format: either uncompressed (counts as list) or compressed (counts as str/bytes)
                if isinstance(seg.get('counts'), list):
                    rle = mask.frPyObjects(seg, h, w)
                else:
                    counts = seg.get('counts', '')
                    rle = {'size': seg.get('size', [h, w]), 'counts': counts.encode('utf-8') if isinstance(counts, str) else counts}
                m = mask.decode(rle)
                if m.ndim == 3:
                    m = np.sum(m, axis=2)
            elif isinstance(seg, np.ndarray):
                m = seg
            else:
                m = np.zeros((h, w), dtype=np.uint8)
        except Exception as e:
            print(f"[Warning] Failed to decode mask for ref_id {ref.get('ref_id', 'unknown')}: {e}")
            m = np.zeros((h, w), dtype=np.uint8)

        if m.shape != (h, w):
            aligned = np.zeros((h, w), dtype=np.uint8)
            h_lim, w_lim = min(m.shape[0], h), min(m.shape[1], w)
            aligned[:h_lim, :w_lim] = m[:h_lim, :w_lim]
            m = aligned

        m = (m > 0).astype(np.uint8)
        area = float(m.sum())

        return {'mask': m, 'area': area}


    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


if __name__ == '__main__':
    refer = REFER(dataset='refcocog', splitBy='google')
    ref_ids = refer.getRefIds()

    ref_ids = refer.getRefIds(split='train')
    print('There are %s training referred objects.' % len(ref_ids))

    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if len(ref['sentences']) < 2:
            continue
        print('The label is %s.' % refer.Cats[ref['category_id']])
        plt.figure()
        refer.showRef(ref, seg_box='box')
        plt.show()

