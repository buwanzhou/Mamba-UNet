import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

def parse_args():
    parser = argparse.ArgumentParser(description='Test Mamba-UNet on ACDC dataset')
    parser.add_argument('--root_path', type=str,
                        default='../data/ACDC', help='Root path of the dataset')
    parser.add_argument('--exp', type=str,
                        default='ACDC/VIM', help='Experiment name')
    parser.add_argument('--model', type=str,
                        default='mambaunet', help='Model name')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='Number of classes')
    parser.add_argument('--labeled_num', type=int, default=140,
                        help='Number of labeled data')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/ccw/code/Mamba-UNet/model/ACDC/VIM_140_labeled/mambaunet/mambaunet_best_model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--save_results', type=bool, default=True,
                        help='Whether to save visualization results')
    parser.add_argument('--save_dir', type=str,
                        default='/home/ccw/code/Mamba-UNet/model/ACDC/VIM_140_labeled/mambaunet/predictions',
                        help='Directory to save results')
    return parser.parse_args()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    
    if FLAGS.save_results:
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
        
        # 可视化中间切片
        mid_slice = image.shape[0] // 2
        visualize_slice(image[mid_slice], label[mid_slice], prediction[mid_slice], 
                       test_save_path + case + "_mid_slice.png")
    
    return first_metric, second_metric, third_metric


def visualize_slice(image, label, prediction, save_path):
    """
    可视化单个切片的图像、标签和预测结果
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # 真实标签
    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='jet')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # 预测结果
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='jet')
    plt.title("Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    
    # 使用命令行参数中指定的模型路径
    snapshot_path = os.path.dirname(FLAGS.checkpoint)
    test_save_path = FLAGS.save_dir
    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    
    save_mode_path = FLAGS.checkpoint
    print("Loading model from:", save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("Model loaded successfully")
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    
    dice_total = 0.0
    hd95_total = 0.0
    asd_total = 0.0
    
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        
        # 计算每个类别的指标
        first_dice, first_hd95, first_asd = first_metric
        second_dice, second_hd95, second_asd = second_metric
        third_dice, third_hd95, third_asd = third_metric
        
        # 累加指标
        first_total += first_dice
        second_total += second_dice
        third_total += third_dice
        
        dice_total += (first_dice + second_dice + third_dice) / 3
        hd95_total += (first_hd95 + second_hd95 + third_hd95) / 3
        asd_total += (first_asd + second_asd + third_asd) / 3
    
    # 计算平均指标
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    
    avg_dice = dice_total / len(image_list)
    avg_hd95 = hd95_total / len(image_list)
    avg_asd = asd_total / len(image_list)
    
    # 打印结果
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    print(f"Class 1 Dice: {avg_metric[0]:.4f}")
    print(f"Class 2 Dice: {avg_metric[1]:.4f}")
    print(f"Class 3 Dice: {avg_metric[2]:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average HD95: {avg_hd95:.4f}")
    print(f"Average ASD: {avg_asd:.4f}")
    print("="*50 + "\n")
    
    # 保存结果到文本文件
    results_path = os.path.join(test_save_path, "test_results.txt")
    with open(results_path, "w") as f:
        f.write("Test Results:\n")
        f.write("="*50 + "\n")
        f.write(f"Class 1 Dice: {avg_metric[0]:.4f}\n")
        f.write(f"Class 2 Dice: {avg_metric[1]:.4f}\n")
        f.write(f"Class 3 Dice: {avg_metric[2]:.4f}\n")
        f.write(f"Average Dice: {avg_dice:.4f}\n")
        f.write(f"Average HD95: {avg_hd95:.4f}\n")
        f.write(f"Average ASD: {avg_asd:.4f}\n")
        f.write("="*50 + "\n")
    
    print(f"Results saved to {results_path}")
    
    return avg_metric


if __name__ == '__main__':
    FLAGS = parse_args()
    metric = Inference(FLAGS)
