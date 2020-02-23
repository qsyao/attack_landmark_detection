import argparse
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader

from network import UNet
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from utils import to_Image, voting
from attack import FGSMAttack

def L1Loss(pred, gt, mask=None):
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()
    # return distence.mean()

class Tester(object):
    def __init__(self, logger, config, net=None, net2=None, output_file=None,\
            attacker=None):
        dataset_1 = Cephalometric(config['dataset_pth'], 'Test1')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
        
        dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
        
        self.Radius = dataset_1.Radius
        
        self.model = net 
        self.net2 = net2

        self.evaluater = Evaluater(logger, dataset_1.size, \
            dataset_1.original_size)
        self.logger = logger

        self.attacker = attacker

    def debug(self, net=None):
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        for img, mask, offset_y, offset_x, gt, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda()
            heatmap, regression_y, regression_x = self.model(img)

            regression_x, regression_y = regression_x*mask, regression_y*mask
            to_Image(heatmap[0][0], show="heatmap", normalize=False)
            to_Image(mask[0][0], show="mask")
            to_Image(regression_x[0][0], show="regression_x", normalize=True)
            to_Image(offset_x[0][0], show="offset_x", normalize=True)
            to_Image(regression_y[0][0], show="regression_y", normalize=True)
            to_Image(offset_y[0][0], show="offset_y", normalize=True)

            import ipdb; ipdb.set_trace()

    def test(self, net=None):
        self.evaluater.reset()
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        for img, mask, offset_y, offset_x, gt, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda()

            heatmap, regression_y, regression_x = self.model(img)
            heatmap, _, __ = self.net2(img)

            pred_landmark = voting(heatmap, regression_y, regression_x, self.Radius, landmark_list, mask)
            gg = self.evaluater.record(pred_landmark, landmark_list)
            if gg is not None:
                import ipdb; ipdb.set_trace()
                to_Image(heatmap[0][gg], show="false_heatmap", normalize=False)
                to_Image(mask[0][gg], show="false_mask")

        self.evaluater.cal_metrics()
    
    def attack(self):
        self.evaluater.reset()
        assert(hasattr(self, 'attacker'))
        for img, mask, offset_y, offset_x, gt, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda()

            adv_img = attacker.FGSM_Untarget_Heatmap(img, debug=True)

            heatmap, regression_y, regression_x = self.model(adv_img)
            # heatmap, _, __ = self.net2(adv_img)

            pred_landmark = voting(heatmap, regression_y, regression_x, self.Radius, landmark_list, mask)
            self.evaluater.record(pred_landmark, landmark_list)
        
        self.evaluater.cal_metrics()

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='test', help="position of the output dir")
    parser.add_argument("--debug", default='', help="position of the output dir")
    parser.add_argument("--attack", default='', help="position of the output dir")
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
    net_logic = UNet(1, config['num_landmarks']).cuda()
    logger.info("Loading checkpoints from epoch {}".format(config['last_epoch']))
    checkpoints = torch.load(os.path.join(config['runs_dir'], \
                        "model_epoch_{}.pth".format(config['last_epoch'])))
    net.load_state_dict(checkpoints)
    # checkpoints = torch.load(os.path.join('runs/only_logic', \
    #                     "model_epoch_{}.pth".format(89)))
    net_logic.load_state_dict(checkpoints)
    
    attacker = FGSMAttack(net_logic, BCELoss(), 8)   
    tester = Tester(logger, config, net, net_logic, output_file, attacker)
    if args.debug != '':
        tester.debug()

    if args.attack == '':
        tester.test()
    else:
        tester.attack()
