import os
import sys
import argparse
import torch
import time
import numpy as np
import cv2
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model


img_path1='/home/hhylqb/lby/awesome-semantic-segmentation-pytorch-master/datasets/ade/ADEChallengeData2016/images/validation/ADE_val_00000009.jpg'
img_path2='/home/hhylqb/lby/awesome-semantic-segmentation-pytorch-master/tests/44.png'
img_path3='/home/lby/segmentation/awesome-semantic-segmentation-pytorch-master/tests/3_1.jpg'
img_path4='/home/lby/segmentation/awesome-semantic-segmentation-pytorch-master/tests/3_1.jpg'

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='psp_resnet50_ade',#default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='ade20k', choices=['pascal_voc/pascal_aug/ade20k/citys'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default=img_path3,
                    help='path to the input picture')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
args = parser.parse_args()


def demo(config):
    print('gpu flag:',torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert('RGB')
    image1 = Image.open(img_path4).convert('RGB')

    images = transform(image).unsqueeze(0).to(device)
    images1 = transform(image1).unsqueeze(0).to(device)

    image_mul=torch.cat((images,images1),0)
    '''print the image's size'''
    print("image_mul type is:",type(image_mul))
    print("the image's size is:", image_mul.size())
    image_mul = image_mul.to("cuda")
    #print("images[0]:",images[0].size())

    model = get_model(args.model, pretrained=True, root=args.save_folder).to(device)
    print('Finished loading model!')

    for i in range(1):
        model.eval()
        Time_start = time.time()
        with torch.no_grad():
            #output = model(images)

            output = model(image_mul)
            print(output[0].size())
            output_0=output[0][:1]
            output_1 = output[0][1:]
            print(output_0.size())

        #pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        pred = torch.argmax(output_0, 1).squeeze(0).cpu().data.numpy()
        pred1 = torch.argmax(output_1, 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, args.dataset)
        mask1 = get_color_pallete(pred1, args.dataset)
        mask_rgb = mask.convert('RGB')
        img = cv2.cvtColor(np.asarray(mask_rgb),cv2.COLOR_RGB2BGR)
        cv2.imshow("img",img)
        print("img.size:",img.shape)

        Time_end=time.time()
        print('predict time=',Time_end-Time_start)
        outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '_1027_bisenet206epochs_mul_input3.png'
        outname1 = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '_1027_bisenet206epochs_mul_input44.png'
        mask.save(os.path.join(args.outdir, outname))
        mask1.save(os.path.join(args.outdir, outname1))

        key=cv2.waitKey(0)
        if key==32:
            break

if __name__ == '__main__':
    demo(args)
