import argparse
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import yaml
import yamlloader

from network import UNet
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import cal_metrics
from utils import to_Image, show_heatmap

class Tester(object):
    def __init__(self, config, net=None, output_file=None):
        dataset_1 = Cephalometric(config['dataset_pth'], 'Train')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=True, num_workers=config['num_workers'])
        
        dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
                                shuffle=True, num_workers=config['num_workers'])
        
        self.size = dataset_1.size
        self.original_size = dataset_1.original_size
        self.Radius = dataset_1.Radius
        
        self.model = net  

    def test(self, net=None):
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        for img, mask, offset_y, offset_x, gt, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda()
            heatmap, regression_y, regression_x = self.model(img)
            regression_x, regression_y = regression_x*mask, regression_y*mask
            
            # to_Image(pred[0][0], show="pred")
            # to_Image(mask[0][0], show="mask")
            to_Image(regression_x[0][0], show="regression_x", normalize=True)
            to_Image(offset_x[0][0], show="offset_x", normalize=True)
            to_Image(regression_y[0][0], show="regression_y", normalize=True)
            to_Image(offset_y[0][0], show="offset_y", normalize=True)
            # show_heatmap(pred, Radius=self.Radius)
            import ipdb; ipdb.set_trace()
            metrics = cal_metrics(pred, landmark_list, \
                self.size, self.original_size)

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='test', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--checkpoint_file", default="", help="default configs")
    parser.add_argument("--output_file", default="", help="default configs")
    args = parser.parse_args()

    with open(os.path.join("runs", args.tag, args.config_file), "r") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()

    if args.output_file == "":
        output_file = os.path.join("runs", args.tag, args.tag+"_result.csv")
    else:
        output_file = args.output_file
    
    # Load model
    net = UNet(1, config['num_landmarks']).cuda()
    checkpoints = torch.load(os.path.join(config['runs_dir'], \
                        "model_epoch_{}.pth".format(79)))
    net.load_state_dict(checkpoints)
    
    tester = Tester(config, net, output_file)
    tester.test()
