"""
Dataset loaders for RRSIS_SAM3.

Supports three datasets:
    1. RRSIS-D: 17402 triplets, 20 categories, 800×800
    2. RRSIS-HR: 2650 triplets, 7 categories, 1024×1024
    3. RefSegRS: 4402 triplets, 512×512

All images are resized to 504×504 (divisible by 14 for SAM3's ViT patch size).
No BEiT-3 tokenizer needed — SAM3 handles tokenization internally.
"""

import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF

from refer.refer import REFER
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transforms import TargetAwareZoomCrop, RandomHorizontalFlip


def safe_load_image(image_dir, file_name, dataset_root=None, cache=None):
    """
    Safely load an image with multiple fallback strategies:
    1. Direct match: image_dir / file_name
    2. Extension variations: .jpg, .png, .tif, .jpeg, etc.
    3. Recursive lookup via cache if file is in a nested subdirectory
    """
    # 1. Direct path
    img_path = os.path.join(image_dir, file_name)
    if os.path.isfile(img_path):
        return Image.open(img_path).convert('RGB')

    # 2. Check alternative extensions
    base, _ = os.path.splitext(file_name)
    for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp', '.JPG', '.PNG', '.JPEG']:
        cand = os.path.join(image_dir, base + ext)
        if os.path.isfile(cand):
            return Image.open(cand).convert('RGB')

    # 3. Dynamic recursive search using cache
    if cache is not None:
        if not cache:
            # Populate cache by scanning image_dir and dataset_root
            search_dirs = [image_dir]
            if dataset_root and os.path.exists(dataset_root) and dataset_root != image_dir:
                search_dirs.append(dataset_root)
            for sdir in search_dirs:
                if os.path.exists(sdir):
                    for root, _, files in os.walk(sdir):
                        for f in files:
                            fl = f.lower()
                            if fl.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                                f_base, _ = os.path.splitext(f)
                                if f not in cache:
                                    cache[f] = os.path.join(root, f)
                                if f_base not in cache:
                                    cache[f_base] = os.path.join(root, f)

        # Lookup by exact filename
        if file_name in cache:
            return Image.open(cache[file_name]).convert('RGB')
        # Lookup by base name without extension
        if base in cache:
            return Image.open(cache[base]).convert('RGB')

    raise FileNotFoundError(f"Image '{file_name}' not found in '{image_dir}' or subdirectories.")


# ============================================================
# RRSIS-D Dataset (uses REFER API)
# ============================================================
class RRSISDDataset(data.Dataset):
    """
    RRSIS-D dataset loader.

    Built on the DIOR-RSVG dataset with 20 object categories.
    12181 train / 1740 val / 3481 test triplets.
    """

    def __init__(self, args, split='train', eval_mode=False):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size
        self.return_orig_mask = getattr(args, 'eval_native_size', False) and eval_mode

        # Load REFER annotations
        self.refer = REFER(args.data_root, 'rrsis_d', args.splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        # Build captions
        self.captions = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            caption_for_ref = [el['raw'] for el in ref['sentences']]
            self.captions.append(caption_for_ref)

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # Zoom-In Crop & Horizontal Flip
        is_train_mode = split in ['train', 'trainval', 'train_val']
        self.zoom_crop = TargetAwareZoomCrop(zoom_prob=0.5, small_threshold=0.1) if is_train_mode else None
        self.hflip = RandomHorizontalFlip(flip_prob=0.5) if is_train_mode else None

        # Random augmentation for training
        if split == 'train':
            num_to_mask = int(len(ref_ids) * 0.2)
            self.images_to_mask = set(random.sample(ref_ids, num_to_mask))
        else:
            self.images_to_mask = set()

        self._img_cache = {}
        print(f"[RRSIS-D] Loaded {len(self.ref_ids)} samples for {split}")

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        # Load image with fallback resolution
        img = safe_load_image(self.refer.IMAGE_DIR, this_img['file_name'],
                              dataset_root=os.path.dirname(self.refer.DATA_DIR),
                              cache=self._img_cache)

        # Random occlusion augmentation
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = self._add_random_boxes(img)

        # Load mask
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = (ref_mask > 0).astype(np.uint8)

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]  # Return all captions for eval
        else:
            caption = random.choice(self.captions[index])

        # Target-aware zoom crop (safely preserves anchors on directional queries)
        if self.zoom_crop is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil = self.zoom_crop(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Random horizontal reflection with relation inversion
        if self.hflip is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil, caption = self.hflip(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Resize image and mask
        img = self.img_transform(img)
        orig_mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(orig_mask.clone(), [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.return_orig_mask:
            return img, mask, caption, orig_mask
        return img, mask, caption

    @staticmethod
    def _add_random_boxes(img, min_num=2, max_num=6, box_size=16):
        img_np = np.asarray(img).copy()
        img_h, img_w = img_np.shape[:2]
        num = random.randint(min_num, max_num)
        for _ in range(num):
            y = random.randint(0, max(0, img_h - box_size))
            x = random.randint(0, max(0, img_w - box_size))
            img_np[y:y+box_size, x:x+box_size] = 0
        return Image.fromarray(img_np, 'RGB')


# ============================================================
# RRSIS-HR Dataset (uses REFER API)
# ============================================================
class RRSISHRDataset(data.Dataset):
    """
    RRSIS-HR dataset loader.

    Very high resolution RS images with longer language expressions.
    2118 train / 268 val / 264 test triplets.
    7 object categories, 1024×1024 images.
    """

    def __init__(self, args, split='train', eval_mode=False):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size
        self.return_orig_mask = getattr(args, 'eval_native_size', False) and eval_mode

        # Load REFER annotations
        self.refer = REFER(args.data_root, 'rrsis_hr', args.splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        # Build captions
        self.captions = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            caption_for_ref = [el['raw'] for el in ref['sentences']]
            self.captions.append(caption_for_ref)

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # Zoom-In Crop & Horizontal Flip
        is_train_mode = split in ['train', 'trainval', 'train_val']
        self.zoom_crop = TargetAwareZoomCrop(zoom_prob=0.5, small_threshold=0.1) if is_train_mode else None
        self.hflip = RandomHorizontalFlip(flip_prob=0.5) if is_train_mode else None

        self._img_cache = {}
        print(f"[RRSIS-HR] Loaded {len(self.ref_ids)} samples for {split}")

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        # Load image with fallback resolution
        img = safe_load_image(self.refer.IMAGE_DIR, this_img['file_name'],
                              dataset_root=os.path.dirname(self.refer.DATA_DIR),
                              cache=self._img_cache)

        # Load mask
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = (ref_mask > 0).astype(np.uint8)

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]
        else:
            caption = random.choice(self.captions[index])

        # Target-aware zoom crop (safely preserves anchors on directional queries)
        if self.zoom_crop is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil = self.zoom_crop(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Random horizontal reflection with relation inversion
        if self.hflip is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil, caption = self.hflip(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Resize
        img = self.img_transform(img)
        orig_mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(orig_mask.clone(), [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.return_orig_mask:
            return img, mask, caption, orig_mask
        return img, mask, caption


# ============================================================
# RISLAD Dataset (uses REFER API)
# ============================================================
class RISLADDataset(data.Dataset):
    """
    RISLAD dataset loader.
    """

    def __init__(self, args, split='train', eval_mode=False):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size
        self.return_orig_mask = getattr(args, 'eval_native_size', False) and eval_mode

        # Load REFER annotations
        self.refer = REFER(args.data_root, 'rislad', args.splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        # Build captions
        self.captions = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            caption_for_ref = [el['raw'] for el in ref['sentences']]
            self.captions.append(caption_for_ref)

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # Zoom-In Crop & Horizontal Flip
        is_train_mode = split in ['train', 'trainval', 'train_val']
        self.zoom_crop = TargetAwareZoomCrop(zoom_prob=0.5, small_threshold=0.1) if is_train_mode else None
        self.hflip = RandomHorizontalFlip(flip_prob=0.5) if is_train_mode else None

        # Random augmentation for training
        if split == 'train':
            num_to_mask = int(len(ref_ids) * 0.2)
            self.images_to_mask = set(random.sample(ref_ids, num_to_mask))
        else:
            self.images_to_mask = set()

        self._img_cache = {}
        print(f"[RISLAD] Loaded {len(self.ref_ids)} samples for {split}")

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        # Load image with fallback resolution
        img = safe_load_image(self.refer.IMAGE_DIR, this_img['file_name'],
                              dataset_root=os.path.dirname(self.refer.DATA_DIR),
                              cache=self._img_cache)

        # Random occlusion augmentation
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = self._add_random_boxes(img)

        # Load mask
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = (ref_mask > 0).astype(np.uint8)

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]  # Return all captions for eval
        else:
            caption = random.choice(self.captions[index])

        # Target-aware zoom crop (safely preserves anchors on directional queries)
        if self.zoom_crop is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil = self.zoom_crop(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Random horizontal reflection with relation inversion
        if self.hflip is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil, caption = self.hflip(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Resize image and mask
        img = self.img_transform(img)
        orig_mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(orig_mask.clone(), [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.return_orig_mask:
            return img, mask, caption, orig_mask
        return img, mask, caption

    @staticmethod
    def _add_random_boxes(img, min_num=20, max_num=60, box_size=32):
        img_np = np.asarray(img).copy()
        img_size = img_np.shape[1]
        num = random.randint(min_num, max_num)
        for _ in range(num):
            y = random.randint(0, img_size - box_size)
            x = random.randint(0, img_size - box_size)
            img_np[y:y+box_size, x:x+box_size] = 0
        return Image.fromarray(img_np, 'RGB')


# ============================================================
# RefSegRS Dataset (file-based, not REFER API)
# ============================================================
class RefSegRSDataset(data.Dataset):
    """
    RefSegRS dataset loader.

    First remote sensing referring segmentation dataset.
    2172 train / 413 val / 1817 test, 512×512 images.
    """

    def __init__(self, args, split='train', eval_mode=False, data_root=None):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size
        self.return_orig_mask = getattr(args, 'eval_native_size', False) and eval_mode
        
        # Robust data_root resolution for RefSegRS
        target_root = data_root or args.data_root
        cand_roots = [
            target_root,
            os.path.join(target_root, 'RefSegRS'),
            os.path.join(target_root, 'refsegrs'),
            os.path.join(target_root, 'refseg-rs'),
            os.path.join('..', target_root),
            os.path.join('..', 'data', 'RefSegRS'),
            os.path.join('..', 'RefSegRS'),
            './RefSegRS',
            '../RefSegRS',
            './data',
            '../data',
        ]
        self.data_root = target_root
        for cr in cand_roots:
            if cr and os.path.exists(cr):
                # Check if it directly contains the split text file or an images dir
                if (os.path.exists(os.path.join(cr, f'output_phrase_{split}.txt')) or 
                    os.path.exists(os.path.join(cr, f'{split}.txt')) or
                    os.path.exists(os.path.join(cr, 'images'))):
                    self.data_root = cr
                    break

        # Load data from text files (Auto-combine train + val for full SOTA coverage)
        if split in ['train', 'trainval', 'train_val']:
            imgs_tr, lbls_tr, sents_tr = self._build_batches('train')
            imgs_va, lbls_va, sents_va = self._build_batches('val')
            self.imgs = imgs_tr + imgs_va
            self.labels = lbls_tr + lbls_va
            self.sentences = sents_tr + sents_va
            print(f"[RefSegRS] Auto-merged train ({len(imgs_tr)}) + val ({len(imgs_va)}) -> Total {len(self.imgs)} samples for SOTA training!")
        else:
            self.imgs, self.labels, self.sentences = self._build_batches(split)

        # Build caption lists
        self.captions = []
        for sent in self.sentences:
            self.captions.append([sent.strip()])

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # Zoom-In Crop & Horizontal Flip
        self.zoom_crop = TargetAwareZoomCrop(zoom_prob=0.5, small_threshold=0.1) if split in ['train', 'trainval', 'train_val'] else None
        self.hflip = RandomHorizontalFlip(flip_prob=0.5) if split in ['train', 'trainval', 'train_val'] else None

        print(f"[RefSegRS] Loaded {len(self.imgs)} samples for {split}")

    def _build_batches(self, split):
        """Load image/mask paths and captions from text files."""

        im_dir = os.path.join(self.data_root, 'images')
        if not os.path.exists(im_dir):
            for cand in ['Images', 'JPEGImages', 'img']:
                cand_path = os.path.join(self.data_root, cand)
                if os.path.exists(cand_path):
                    im_dir = cand_path
                    break

        seg_dir = os.path.join(self.data_root, 'masks')
        if not os.path.exists(seg_dir):
            for cand in ['Masks', 'labels', 'Labels', 'annotations']:
                cand_path = os.path.join(self.data_root, cand)
                if os.path.exists(cand_path):
                    seg_dir = cand_path
                    break

        split_files = [
            f'output_phrase_{split}.txt',
            f'output_phrases_{split}.txt',
            f'{split}.txt',
            f'output_{split}.txt',
        ]
        set_file = None
        for sf in split_files:
            candidate = os.path.join(self.data_root, sf)
            if os.path.exists(candidate):
                set_file = candidate
                break

        if set_file is None:
            # Fallback recursive search if files are in a nested directory
            for sf in split_files:
                for root, _, files in os.walk(self.data_root):
                    if sf in files:
                        set_file = os.path.join(root, sf)
                        # Also align im_dir and seg_dir to this root if they exist there
                        if os.path.exists(os.path.join(root, 'images')):
                            im_dir = os.path.join(root, 'images')
                        if os.path.exists(os.path.join(root, 'masks')):
                            seg_dir = os.path.join(root, 'masks')
                        break
                if set_file:
                    break

        all_imgs, all_labels, all_sentences = [], [], []
        if set_file and os.path.exists(set_file):
            with open(set_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(' ')
                    img_name = parts[0]
                    sentence = ' '.join(parts[1:])

                    # Resolve image path with extension fallback
                    img_p = os.path.join(im_dir, img_name + '.tif')
                    if not os.path.exists(img_p):
                        for ext in ['.png', '.jpg', '.jpeg', '']:
                            cand = os.path.join(im_dir, img_name + ext)
                            if os.path.exists(cand):
                                img_p = cand
                                break

                    # Resolve label path with extension fallback
                    lbl_p = os.path.join(seg_dir, img_name + '.tif')
                    if not os.path.exists(lbl_p):
                        for ext in ['.png', '.jpg', '.jpeg', '']:
                            cand = os.path.join(seg_dir, img_name + ext)
                            if os.path.exists(cand):
                                lbl_p = cand
                                break

                    all_imgs.append(img_p)
                    all_labels.append(lbl_p)
                    all_sentences.append(sentence)

        if len(all_imgs) == 0:
            print(f"⚠️ [WARNING] Could not find split annotations for '{split}' in '{self.data_root}'.")
            print(f"    Looked for files: {split_files}")
            print(f"    Please check the directory path passed to --data_root.")

        return all_imgs, all_labels, all_sentences

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img = Image.open(self.imgs[index]).convert('RGB')

        # Load mask
        label_mask = cv2.imread(self.labels[index], 2)
        ref_mask = np.array(label_mask) > 50
        annot = np.zeros(ref_mask.shape, dtype=np.uint8)
        annot[ref_mask == 1] = 1

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]
        else:
            caption = self.captions[index][0]

        # Target-aware zoom crop (safely preserves anchors on directional queries)
        if self.zoom_crop is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil = self.zoom_crop(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Random horizontal reflection with relation inversion
        if self.hflip is not None:
            mask_pil = Image.fromarray(annot)
            img, mask_pil, caption = self.hflip(img, mask_pil, caption=caption)
            annot = np.array(mask_pil)

        # Resize
        img = self.img_transform(img)
        orig_mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(orig_mask.clone(), [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.return_orig_mask:
            return img, mask, caption, orig_mask
        return img, mask, caption


# ============================================================
# Dataset Factory
# ============================================================
def get_dataset(args, split='train', eval_mode=False):
    """
    Get the appropriate dataset based on args.dataset.

    Args:
        args: Parsed arguments with dataset name and paths
        split: 'train', 'val', or 'test'
        eval_mode: If True, return all captions per sample

    Returns:
        Dataset instance
    """
    dataset_map = {
        # RRSIS-D
        'rrsis_d': RRSISDDataset,
        'rrsis-d': RRSISDDataset,
        'rrsisd': RRSISDDataset,
        # RRSIS-HR
        'rrsis_hr': RRSISHRDataset,
        'rrsis-hr': RRSISHRDataset,
        'rrsishr': RRSISHRDataset,
        # RIS-LAD
        'rislad': RISLADDataset,
        'ris_lad': RISLADDataset,
        'ris-lad': RISLADDataset,
        # RefSegRS
        'refsegrs': RefSegRSDataset,
        'refseg-rs': RefSegRSDataset,
        'refseg_rs': RefSegRSDataset,
    }

    dataset_key = args.dataset.lower()
    if dataset_key not in dataset_map:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from {list(dataset_map.keys())}")

    dataset_class = dataset_map[dataset_key]
    return dataset_class(args, split=split, eval_mode=eval_mode)


def collate_fn(batch):
    """
    Custom collate function for RRSIS datasets.

    Handles variable-length captions alongside tensor data and optional native ground truth masks.
    """
    if len(batch[0]) == 4:
        images, masks, captions, orig_masks = zip(*batch)
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        try:
            orig_masks = torch.stack(orig_masks, dim=0)
        except Exception:
            orig_masks = list(orig_masks)
        return images, masks, list(captions), orig_masks

    images, masks, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks, list(captions)
