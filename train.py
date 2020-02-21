import argparse
import datetime
import os
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from network import UNet
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from test import Tester

def L1Loss(pred, gt, mask=None):
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        pass
        # distence = distence * mask
    # return distence.sum() / mask.sum()
    return distence.mean()

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    args = parser.parse_args()

    # Load yaml config file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    runs_dir = "./runs/" + tag
    runs_path = Path(runs_dir)
    config['runs_dir'] = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)

    dataset = Cephalometric(config['dataset_pth'], 'Train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'])
    
    net = UNet(1, config['num_landmarks'])
    net = net.cuda()
    logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])

    # loss
    loss_logic_fn = BCELoss()
    loss_regression_fn = L1Loss

    # Tester
    tester = Tester(config)

    for epoch in range(config['num_epochs']):
        logic_loss_list = list()
        regression_loss_list = list()
        net.train()
        for img, mask, offset_y, offset_x, gt, landmark_list in tqdm(dataloader):
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda()
            heatmap, regression_y, regression_x = net(img)
            
            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss =  regression_loss_x + regression_loss_y# + logic_loss
            loss_regression = regression_loss_y + regression_loss_x

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(logic_loss.cpu().item())
            regression_loss_list.append(loss_regression.cpu().item())            
        logger.info("Epoch {} Training logic loss {} regression loss {}".\
            format(epoch, sum(logic_loss_list) / dataset.__len__(), \
                sum(regression_loss_list) / dataset.__len__()))

        logic_loss_list = list()
        regression_loss_list = list()
        net.eval()
        for img, mask, offset_y, offset_x, gt, landmark_list in tqdm(tester.dataloader_1):
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda()
            with torch.no_grad():
                heatmap, regression_y, regression_x = net(img)
                
                logic_loss = loss_logic_fn(heatmap, mask)
                regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
                regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

                loss = logic_loss + regression_loss_x + regression_loss_y
                loss_regression = regression_loss_y + regression_loss_x

            logic_loss_list.append(logic_loss.cpu().item())
            regression_loss_list.append(loss_regression.cpu().item()) 
        logger.info("Epoch {} Testing logic loss {} regression loss {}".\
            format(epoch, sum(logic_loss_list) / dataset.__len__(), \
                sum(regression_loss_list) / dataset.__len__()))
        # save model
        if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
    
        config['last_epoch'] = epoch

    # dump yaml
    with open(runs_dir + "/config.yaml", "w") as f:
        yaml.dump(config, f)
