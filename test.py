import os
import argparse

import torch
import numpy as np
import PIL.Image as pil_image
from interpolation import pil_resize, torch_resize

import torchvision.transforms as T

from models import RDN
from utils import AverageMeter, convert_rgb_to_y, denormalize

from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', type=str, default='/home/rahul/Workspace/GitHub/SR-DEQNET/Data/TestImages/Set5')
    parser.add_argument('--output-dir', type=str, default='Test_Results')

    parser.add_argument('--interp-scale', type=int, default=4)
    parser.add_argument('--interp-method', type=str, default='TORCH',
                        help='choose among PIL, TORCH')

    # Parameters related to Model specification
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)

    parser.add_argument('--use-gpu', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--save-image', type=bool, default=False)

    args = parser.parse_args()

    # Select Device
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print("Device = ", device)

    # Define Output Directory
    output_dir = os.path.join(args.output_dir, os.path.split(args.input_dir)[1],
                                    "{}".format(args.interp_method),
                                    "x{}".format(args.interp_scale))
    log_file_name = os.path.join(output_dir, 'testing_log.txt')

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)                                    

    # Define Model's weight file path
    weights_file = os.path.join('learned_Models',
                                "{}".format(args.interp_method),
                                "x{}".format(args.interp_scale),
                                'best.pth')
    # Instatiate Model
    model = RDN(scale_factor=args.interp_scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    # Load Model weights
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    data_error_avg = AverageMeter()

    with open(log_file_name, 'w') as fp:
        input_dir = args.input_dir
        with os.scandir(input_dir) as entries:
            for i, entry in enumerate(entries):
                if entry.is_file():
                    # Read image file
                    input_file_name = os.path.join(input_dir, entry.name)
                    img_hr = pil_image.open(input_file_name).convert('RGB')

                    # Prepare it for Downsampling and Upsampling
                    img_height, img_width = img_hr.height, img_hr.width
                    img_width = (img_width // args.interp_scale) * args.interp_scale
                    img_height = (img_height // args.interp_scale) * args.interp_scale
                    img_hr = img_hr.resize((img_width, img_height), resample=pil_image.Resampling.BICUBIC)

                    if args.interp_method == 'PIL':
                        img_lr = np.asarray(img_hr.resize((img_width//args.interp_scale, img_height//args.interp_scale), resample=pil_image.Resampling.BICUBIC), dtype=np.float32)
                        img_hr = np.asarray(img_hr, dtype=np.float32)
                    elif args.interp_method == 'TORCH':
                        img_hr = np.asarray(img_hr, dtype=np.float32)
                        img_lr = torch_resize(img_hr, (1/args.interp_scale), 3)
                        img_lr = np.asarray(img_lr, dtype='float32')
                        img_hr = np.asarray(img_hr, dtype='float32')

                        # Instantiate TorchVision Resize
                        # torch_resize = T.Resize((img_height//args.interp_scale, img_width//args.interp_scale), interpolation=T.InterpolationMode.BICUBIC)
                        # img_hr = torch.from_numpy(np.asarray(img_hr).transpose(2,0,1)).unsqueeze(dim=0)
                        # img_lr = torch_resize(img_hr)
                        # img_lr = np.asarray(img_lr.squeeze().permute(1,2,0), dtype='float32')
                        # img_hr = np.asarray(img_hr.squeeze().permute(1,2,0), dtype='float32')
                    else:
                        raise Exception("Unsupported Interpolation Methods!")

                    # Clamp the interpolated image to range [0, 255] because Bicubic can result into out of range pixel value
                    img_lr = img_lr.clip(min=0, max=255)

                    # Convert the input to Torch Tensor
                    img_lr = torch.from_numpy(img_lr.transpose(2,0,1)).unsqueeze(dim=0).to(device)/255.0
                    img_hr = torch.from_numpy(img_hr.transpose(2,0,1)).unsqueeze(dim=0).to(device)/255.0
                    
                    with torch.no_grad():
                        img_sr = model(img_lr)

                    # Calculate Measurement Consistency Error
                    hr_y = convert_rgb_to_y(img_hr*255.0)
                    sr_y = convert_rgb_to_y(img_sr*255.0)
                    lr_y = convert_rgb_to_y(img_lr*255.0)
                    if args.interp_method == 'PIL':
                        sr_y = sr_y.detach().cpu().numpy()
                        lr_y = lr_y.detach().cpu().numpy()
                        data_error = np.linalg.norm((pil_resize(sr_y, scale=(1/args.interp_scale), order=3) - lr_y).flatten())
                    else:
                        sr_y = sr_y.detach().cpu().numpy()
                        lr_y = lr_y.detach().cpu().numpy()
                        data_error = np.linalg.norm((torch_resize(sr_y, scale=(1/args.interp_scale), order=3) - lr_y).flatten())

                        # data_error = np.linalg.norm((torch_resize(sr_y.unsqueeze(dim=0).unsqueeze(dim=0)).detach().cpu().numpy() - lr_y.detach().cpu().numpy()).flatten())
                        # sr_y = sr_y.detach().cpu().numpy()
                        # lr_y = lr_y.detach().cpu().numpy()
                        
                    hr_y = hr_y.detach().cpu().numpy()

                    # # Plot the images
                    # plt.figure()
                    # plt.imshow(hr_y, cmap='gray'); plt.colorbar(); plt.title("HR")
                    # plt.figure()
                    # plt.imshow(lr_y, cmap='gray'); plt.colorbar(); plt.title("LR")
                    # plt.figure()
                    # plt.imshow(sr_y, cmap='gray'); plt.colorbar(); plt.title("SR")
                    # plt.show()

                    # Calculate PSNR and SSIM
                    sr_y = convert_rgb_to_y(denormalize(img_sr))
                    hr_y = convert_rgb_to_y(denormalize(img_hr))
                    sr_y = sr_y[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                    hr_y = hr_y[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                    sr_y = sr_y.detach().cpu().numpy()
                    hr_y = hr_y.detach().cpu().numpy()

                    psnr = calc_psnr(hr_y, sr_y, data_range=255.)
                    ssim = calc_ssim(hr_y, sr_y, data_range=255.)

                    psnr_avg.update(psnr)
                    ssim_avg.update(ssim)
                    data_error_avg.update(data_error)

                    print("Image File Name:                         ", entry.name, file=fp)
                    print("Data Consistency:                        {:.8e}".format(data_error), file=fp)
                    print("Output PSNR:                             {:.4f} & SSIM:{:.4f}".format(psnr, ssim), file=fp)
                    print('Avg PSNR:                                {:.4f} & SSIM: {:.4f}'.format(psnr_avg.avg, ssim_avg.avg), file=fp)
                    print("Avg Data Consistency:                    {:8e}".format(data_error_avg.avg), file=fp)
                    print("\n", file=fp)


    fp.close()   

