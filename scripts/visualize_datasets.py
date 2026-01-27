#!/usr/bin/env python3
import os
import glob
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def find_first_h5(pattern):
    paths = sorted(glob.glob(pattern))
    return paths[0] if paths else None


def load_h5(path):
    with h5py.File(path, 'r') as f:
        img = np.array(f['image'])
        lab = np.array(f['label'])
    return img, lab


def save_overlay(img, lab, outpath, title=None):
    img2 = img.astype(np.float32)
    if img2.max() > img2.min():
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    # if 3D slice provided, squeeze
    if img2.ndim == 3:
        img2 = img2.squeeze()
    if lab.ndim == 3:
        lab = lab.squeeze()

    plt.figure(figsize=(6,6))
    plt.imshow(img2, cmap='gray')
    # color overlay: map labels >0
    unique = np.unique(lab)
    if unique.size > 1:
        cmap = plt.get_cmap('tab10')
        for cls in unique:
            if cls == 0:
                continue
            mask = (lab == cls).astype(np.uint8)
            if mask.sum() == 0:
                continue
            plt.contour(mask, levels=[0.5], colors=[cmap(int(cls) % 10)], linewidths=1)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--acdc', default='data/ACDC', help='ACDC root')
    p.add_argument('--mnm2', default='data/MnM2', help='MnM2 root')
    p.add_argument('--out', default='outputs/vis', help='output dir')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    acdc_h5 = find_first_h5(os.path.join(args.acdc, 'data', '*_slice_*.h5'))
    mnm2_h5 = find_first_h5(os.path.join(args.mnm2, 'data', 'slices', '*.h5'))

    if not acdc_h5 and not mnm2_h5:
        print('No samples found. Check data paths.')
        return

    if acdc_h5:
        img, lab = load_h5(acdc_h5)
        outp = os.path.join(args.out, 'acdc_sample.png')
        save_overlay(img, lab, outp, title=os.path.basename(acdc_h5))
        print('Saved', outp)
    else:
        print('No acdc sample found at', os.path.join(args.acdc, 'data'))

    if mnm2_h5:
        img, lab = load_h5(mnm2_h5)
        outp = os.path.join(args.out, 'mnm2_sample.png')
        save_overlay(img, lab, outp, title=os.path.basename(mnm2_h5))
        print('Saved', outp)
    else:
        print('No mnm2 sample found at', os.path.join(args.mnm2, 'data', 'slices'))


if __name__ == '__main__':
    main()
