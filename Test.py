import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

# deprecated
#from scipy import misc
# so change to
import imageio
from skimage import color
#from skimage.filters import 
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#from lib.HarDMSEG import HarDMSEG
from utils.dataloader import test_dataset
#from CFP_Res2Net import cfpnet_res2net
from collections import OrderedDict
#from pranet import PraNet
from CaraNet import caranet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "TestDataset", "snapshots", "CaraNet-bestCaraNet-best.pth")))



#for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'KvasirCapsule-SEG']:
for _data_name in ['KvasirCapsule-SEG']:
    ##### put ur data_path here #####
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "TestDataset", _data_name))
    #####                       #####
    
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "results", _data_name))
    opt = parser.parse_args()
    model = caranet()
    weights = torch.load(opt.pth_path)
    new_state_dict = OrderedDict()

    for k, v in weights.items():

    
        if 'total_ops' not in k and 'total_params' not in k:
            name = k
            new_state_dict[name] = v
        # print(new_state_dict[k])
        
            # # print(k)
        # fp = open('./log3.txt','a')
        # fp.write(str(k)+'\n')
        # fp.close()
    # print(new_state_dict)
        
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()


    os.makedirs(save_path, exist_ok=True)
    image_root = os.path.join(data_path, "images")
    gt_root = os.path.join(data_path, "masks")
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        #### load ground truth data to show an overlay of the result
        # load mask image
        gt_outline = cv2.imread(os.path.join(gt_root, name))
        gt_outline = cv2.cvtColor(gt_outline, cv2.COLOR_BGR2GRAY)
        # find contours on mask image
        gt_contours = cv2.findContours(gt_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_contours = gt_contours[0] if len(gt_contours) == 2 else gt_contours[1]

        # load orig color data         
        # data loader returns name as png... convert back to jpg
        orig_img_name = name
        if orig_img_name.endswith('.png'):
            orig_img_name = orig_img_name.split('.png')[0] + '.jpg'
        orig_image = cv2.imread(os.path.join(image_root, orig_img_name))
        # transfering that to image
        for c in gt_contours:
            cv2.drawContours(orig_image, [c], -1, (0, 255, 0), thickness=1)

        #cv2.imwrite(os.path.join(save_path , name.split('.jpg')[0] + "_overlay.png"), orig_image)

        # cv2.imshow("result", orig_image)
        # cv2.waitKey()
        # gt_outline = gt_outline.convert("RGBA")
        # result_overlay_image = Image.blend(orig_image, gt_outline, 0.5)
        # cv2.imshow("result", result_overlay_image)
        # cv2.waitKey()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # res = model(image)
        res5,res4,res2,res1 = model(image)
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # create a histogram of the blurred grayscale image
        histogram, bin_edges = np.histogram(res, bins=256, range=(0.0, 1.0))
        plt.plot(bin_edges[0:-1], histogram)
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixels")
        plt.xlim(0, 1.0)
        #plt.show()
        plt.savefig(os.path.join(save_path,name + "_histo.png"))

        # res now is inbetween 0 amd 1
        # use binary thresholding to generate a contour
        res_contour_img = res > 0.35
        # shift to 0-255 8 bit
        res_contour_img = res_contour_img * 255
        res_contour_img = res_contour_img.astype(np.uint8)
        # find contours on mask image
        res_contours = cv2.findContours(res_contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res_contours = res_contours[0] if len(res_contours) == 2 else res_contours[1]

        for c in res_contours:
            cv2.drawContours(orig_image, [c], -1, (255, 0, 0), thickness=1)

        result_overlay_image = orig_image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_path , name.split('.jpg')[0] + "_overlay.png"), result_overlay_image)

        # deprecated!
        #misc.imsave(temp_save_path, res)
        res = res * 255
        res = res.astype(np.uint8)
        imageio.imwrite(os.path.join(save_path, name), res)
        print (" saved image " + name)
        