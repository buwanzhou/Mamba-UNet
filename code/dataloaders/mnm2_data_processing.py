import os
import glob
import h5py
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import random


def find_volume_and_mask(patient_dir):
    # Prefer short-axis ED volume (has ground truth), fallback to ES or CINE
    candidates = []
    for pattern in ("*SA_ED.nii*", "*SA_ES.nii*", "*SA_CINE.nii*", "*LA_ED.nii*", "*LA_ES.nii*"):
        candidates.extend(sorted(glob.glob(os.path.join(patient_dir, pattern))))
    for vol in candidates:
        # construct expected mask name
        if vol.endswith('.nii'):
            mask = vol.replace('.nii', '_gt.nii')
        else:
            mask = vol.replace('.nii.gz', '_gt.nii.gz')
        if os.path.exists(mask):
            return vol, mask
    return None, None


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_mnm2():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dataset_dir = os.path.join(repo_root, 'data', 'MnM2', 'dataset')
    out_dir = os.path.join(repo_root, 'data', 'MnM2')
    data_dir = os.path.join(out_dir, 'data')
    slices_dir = os.path.join(data_dir, 'slices')

    make_dirs(slices_dir)

    patient_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, '*')) if os.path.isdir(d)])
    vol_list = []
    slice_entries = {}

    total_slices = 0
    for p in patient_dirs:
        vol, mask = find_volume_and_mask(p)
        if vol is None:
            print(f"skip {p}, no volume+mask pair found")
            continue
        try:
            itk = sitk.ReadImage(vol)
            img = sitk.GetArrayFromImage(itk)
        except Exception as e:
            print(f"SimpleITK failed to read {vol}, falling back to nibabel: {e}")
            img = nib.load(vol).get_fdata()
            img = np.asarray(img)
        try:
            itk_m = sitk.ReadImage(mask)
            msk = sitk.GetArrayFromImage(itk_m)
        except Exception as e:
            print(f"SimpleITK failed to read {mask}, falling back to nibabel: {e}")
            msk = nib.load(mask).get_fdata()
            msk = np.asarray(msk)
        # normalize
        img = img.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())

        base = os.path.basename(vol).split('.')[0]
        case_name = base

        # save full volume
        vol_path = os.path.join(data_dir, f"{case_name}.h5")
        with h5py.File(vol_path, 'w') as f:
            f.create_dataset('image', data=img, compression='gzip')
            f.create_dataset('label', data=msk, compression='gzip')

        # save slices
        slice_entries[case_name] = []
        for si in range(img.shape[0]):
            slice_name = f"{case_name}_slice_{si}"
            slice_path = os.path.join(slices_dir, f"{slice_name}.h5")
            with h5py.File(slice_path, 'w') as f:
                f.create_dataset('image', data=img[si], compression='gzip')
                f.create_dataset('label', data=msk[si], compression='gzip')
            slice_entries[case_name].append(slice_name)
            total_slices += 1

        vol_list.append(case_name)

    # split train/val (80/20)
    random.seed(42)
    vols = vol_list[:]
    random.shuffle(vols)
    n_train = int(len(vols) * 0.8)
    train_vols = vols[:n_train]
    val_vols = vols[n_train:]

    # write lists
    train_slices = []
    all_slices = []
    for v in vol_list:
        for s in slice_entries.get(v, []):
            all_slices.append(s)
            if v in train_vols:
                train_slices.append(s)

    with open(os.path.join(out_dir, 'train_slices.list'), 'w') as f:
        f.write('\n'.join(train_slices))
    with open(os.path.join(out_dir, 'all_slices.list'), 'w') as f:
        f.write('\n'.join(all_slices))
    with open(os.path.join(out_dir, 'train.list'), 'w') as f:
        f.write('\n'.join(train_vols))
    with open(os.path.join(out_dir, 'val.list'), 'w') as f:
        f.write('\n'.join(val_vols))

    print(f"Processed {len(vol_list)} volumes, {total_slices} slices")
    print(f"Train volumes: {len(train_vols)}, Val volumes: {len(val_vols)}")


if __name__ == '__main__':
    process_mnm2()
