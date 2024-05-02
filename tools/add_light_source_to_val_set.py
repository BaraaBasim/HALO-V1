import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from utils import PatchApplier, PatchTransformer, add_padding
import os
# # 从txt文件加载标签
# def load_labels_from_txt(txt_file):
#     labels = []
#     with open(txt_file, 'r') as file:
#         for line in file:
#             label = line.strip().split(' ')
#             labels.append(label)
#     return labels


def load_images_and_labels(image_directory, label_directory):
    trans = PatchTransformer()
    patch_applier = PatchApplier()
    x = torch.tensor([[354.41794, 483.46146,  47.77419, 667.16058, 632.80481],
        [400.99527, 333.38171, 464.83942,  22.72788, 222.21909],
        [460.20703, 509.20877, 287.70346, 389.35223, 489.27686],
        [455.01001, 227.05267, 338.97299, 644.68121, 265.61053],
        [563.75909, 274.88501, 465.03159, 425.32040, 470.04529],
        [ 30.18596, 201.14371, 267.97318, 362.06223, 677.83301],
        [ 16.10493, 190.95059, 221.42401, 695.26288, 561.59912],
        [661.02026, 368.35538,  28.92921,  35.85462, 358.59457],
        [258.33005,  59.87283, 659.35449, 249.36774, 575.57672],
        [662.68115,  34.01614, 330.28033, 356.41550, 663.87079]], device='cuda:0')

    labels = []
    transform_list = [TF.ToTensor()]
    transformer = TF.Compose(transform_list)
    image_files = os.listdir(image_directory)
    for image_file in image_files:
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            image_path = os.path.join(image_directory, image_file)
            try:
                img = Image.open(image_path).convert("RGB")
                img = transformer(img)
                img = img.unsqueeze(0)
                img = img.cuda()
                # print(img.size())   
                
                
                # Find corresponding label file
                label_file = os.path.splitext(image_file)[0] + ".txt"
                label_path = os.path.join(label_directory, label_file)
                if os.path.exists(label_path):
                    imgWithPatch = img
                    with open(label_path, 'r') as file:
                        for line in file:
                            label = line.strip().split(' ')
                            class_index = int(label[0])
                            x_center, y_center, width, height = map(float, label[1:])
                            targets = torch.tensor([0, 1, x_center, y_center, width, height]).unsqueeze(0)
                            print(label)
                            
                            for i in range(x.size(0)):
                                top_para, bottom_para, left_para, right_para = x[i][0].item(), x[i][1].item(), x[i][2].item(), x[i][3].item();
                                color = x[i][4].item()
                                color = int(color)
                                color = color % 4
                                
                                psf_img = cv2.imread(f'../v0/psf_starlight/{color}.jpg')
                                # print(i, color)
                                psf_img = cv2.cvtColor(psf_img, cv2.COLOR_BGR2RGB)
                                # print(psf_img.shape)
                                psf_img = psf_img[200:500, 200:500, :]
                                # print(psf_img.shape)

                                psf_img = add_padding(psf_img, top_para, bottom_para, left_para, right_para)
                                psf_img = psf_img.astype(np.uint8)
                                psf_img = psf_img / 255
                                psf_img = torch.from_numpy(psf_img).cuda()
                                psf_img = psf_img.permute(2, 0, 1)
                                # im = TF.ToPILImage()(psf_img)
                                # im.save("results/whatisthis.png")
                                # ---------------------------
                                # im = TF.ToPILImage()(psf_img)
                                # im.save("1_psf.png")
                                # ---------------------------
                                patch_tf, patch_mask_tf = trans(psf_img, targets, img)
                                imgWithPatch = patch_applier(imgWithPatch, patch_tf, patch_mask_tf)
                        output_image = TF.ToPILImage()(imgWithPatch[0])
                        output_image.save(f"val_set/{os.path.splitext(image_file)[0]}.png")
                            # labels.append(label)
                else:
                    labels.append(None)  # Handle case when label file is missing
            except Exception as e:
                print(f"Error loading image or label: {image_file}")
                print(e)



# def add_light_source(images, labels):

#     for label in labels:
#         class_index = int(label[0])
#         x_center, y_center, width, height = map(float, label[1:])
#         targets = torch.tensor([0, 1, x_center, y_center, width, height]).unsqueeze(0)
        
#         imgWithPatch = images[counter]
#         counter += 1
#         for i in range(x.size(0)):
#             top_para, bottom_para, left_para, right_para = x[i][0].item(), x[i][1].item(), x[i][2].item(), x[i][3].item();
#             color = x[i][4].item()
#             color = int(color)
#             color = color % 4
#             psf_img = cv2.imread(f'../v0/psf_starlight/{color}.jpg')
#             print(i, color)
#             psf_img = cv2.cvtColor(psf_img, cv2.COLOR_BGR2RGB)
#             print(psf_img.shape)
#             psf_img = psf_img[200:500, 200:500, :]
#             print(psf_img.shape)

#             psf_img = add_padding(psf_img, top_para, bottom_para, left_para, right_para)
#             psf_img = psf_img.astype(np.uint8)
#             psf_img = psf_img / 255
#             psf_img = torch.from_numpy(psf_img).cuda()
#             psf_img = psf_img.permute(2, 0, 1)
#             # im = TF.ToPILImage()(psf_img)
#             # im.save("results/whatisthis.png")
#             # ---------------------------
#             # im = TF.ToPILImage()(psf_img)
#             # im.save("1_psf.png")
#             # ---------------------------
#             patch_tf, patch_mask_tf = trans(psf_img, targets, images[counter])
#             imgWithPatch = patch_applier(imgWithPatch, patch_tf, patch_mask_tf)
#         output_image = TF.ToPILImage()(imgWithPatch[0])
#         output_image.save(f"val_set/{counter}.png")

image_directory = "/home/ps/Baraa/data/nightowls_yolo_format (small)/images/val"

label_directory = "/home/ps/Baraa/data/nightowls_yolo_format (small)/labels/val"

load_images_and_labels(image_directory, label_directory)






