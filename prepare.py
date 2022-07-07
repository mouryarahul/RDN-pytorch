import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from torchvision.transforms import transforms

from matplotlib import pyplot as plt

from interpolation import torch_resize, pil_resize


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_list = sorted(glob.glob('{}/*'.format(args.images_dir)))
    patch_idx = 0

    for i, image_path in enumerate(image_list):
        hr_img = pil_image.open(image_path).convert('RGB')

        # plt.figure()
        # plt.imshow(hr_img)
        # plt.show()

        for hr in transforms.FiveCrop(size=(hr_img.height // 2, hr_img.width // 2))(hr_img):
            hr = np.asarray(hr.resize(((hr.width // args.scale) * args.scale, (hr.height // args.scale) * args.scale), resample=pil_image.Resampling.BICUBIC))
            
            # plt.figure()
            # plt.imshow(hr)
            # plt.show()

            if args.interp_method == 'PIL':
                # lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
                lr = pil_resize(hr, 1/args.scale)
            elif args.interp_method == 'TORCH':
                lr = torch_resize(hr.astype(np.float32), 1/args.scale)
                lr = np.clip(lr, a_min=0, a_max=255).astype(np.uint8)
            else:
                raise Exception("Unsupported Interpolation Method!")

            # hr = np.array(hr)
            # lr = np.array(lr)

            lr_group.create_dataset(str(patch_idx), data=lr)
            hr_group.create_dataset(str(patch_idx), data=hr)

            patch_idx += 1

        print(i, patch_idx, image_path)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.Resampling.BICUBIC)
        hr = np.asarray(hr)

        if args.interp_method == 'PIL':
            # lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
            lr = pil_resize(hr, 1/args.scale)
        elif args.interp_method == 'TORCH':
            lr = torch_resize(hr.astype(np.float32), 1/args.scale)
            lr = np.clip(lr, a_min=0, a_max=255).astype(np.uint8)
        else:
            raise Exception("Unsupported Interpolation Method!")

        # hr = np.array(hr)
        # lr = np.array(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        print(i)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='/media/rahul/DATA/WorkSpace/Multimodal-Data-Processing/Projects/DIV2K/benchmark/Set5/HR') #/media/rahul/DATA/WorkSpace/Multimodal-Data-Processing/Projects/DIV-2K/DIV2K_train_HR
    parser.add_argument('--output-path', type=str, default='datasets_PIL/DIV2K_bicubic_x2.h5')
    parser.add_argument('--interp-method', type=str, default='PIL',
                        help='Choose between PIL or TORCH')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
