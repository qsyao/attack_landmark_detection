import argparse
import csv
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader
import random

from network import UNet, UNet_Pretrained
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from utils import to_Image, voting, visualize, make_dir
from attack import FGSMAttack

def get_MICCAI(miccai):
    landmark_list = dict()
    with open('MICCAI/' + miccai +'.txt', 'r') as f:
        for i in range(19):
            coordinates = f.readline().split(',')
            coordinates = [int(item) for item in coordinates]
            landmark_list[i] = coordinates
    return landmark_list

def L1Loss(pred, gt, mask=None):
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()
    # return distence.mean()

def total_loss(mask, guassian_mask, heatmap, gt_y, gt_x, pred_y, pred_x, lamda, target_list=None):
    b, k, h, w = mask.shape
    logic_loss = BCELoss()
    loss_list = list()
    for i in range(mask.shape[1]):
        channel_loss = 2 * logic_loss(heatmap[0][i], guassian_mask[0][i]) +\
            (L1Loss(pred_y[0][i], gt_y[0][i], mask[0][i]) + L1Loss(pred_x[0][i], gt_x[0][i], mask[0][i]))
        loss_list.append(channel_loss)
    total_loss = np.array(loss_list).sum()
    return total_loss

def total_loss_adaptive(mask, guassian_mask, heatmap, gt_y, gt_x, pred_y, pred_x, lamda, target_list=None):
    b, k, h, w = mask.shape
    logic_loss = BCELoss()
    loss_list = list()
    for i in range(mask.shape[1]):
        channel_loss = 2 * logic_loss(heatmap[0][i], guassian_mask[0][i]) +\
            (L1Loss(pred_y[0][i], gt_y[0][i], mask[0][i]) + L1Loss(pred_x[0][i], gt_x[0][i], mask[0][i]))
        loss_list.append(channel_loss)
    loss_list_mean = torch.tensor(loss_list).mean()
    for i in range(len(loss_list)):
        loss_list[i] *= loss_list[i] / loss_list_mean
    total_loss = np.array(loss_list).sum()
    return total_loss

class Tester(object):
    def __init__(self, logger, config, net=None, tag=None, train="", args=None):
        mode = "Test1" if train == "" else "Train"
        dataset_1 = Cephalometric(config['dataset_pth'], mode)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
        
        # # For anthor Testset, deprecated
        # dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        # self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
        #                         shuffle=False, num_workers=config['num_workers'])
        
        self.Radius = dataset_1.Radius
        self.config = config
        self.args = args
        
        self.model = net 
        
        # Creat evluater to record results
        if args.rand == "":
            self.evaluater = Evaluater(logger, dataset_1.size, \
                dataset_1.original_size)
        else:
            self.evaluater = Evaluater(logger, dataset_1.size, \
                dataset_1.original_size, args.rand)

        self.logger = logger

        self.dataset = dataset_1

        output_file_pth = os.path.join("runs", tag, tag+"_result.csv")
        output_pth = os.path.join("runs", tag)
    
        self.output_dir = make_dir(os.path.join(output_pth, 'results'))
        self.attack_dir = make_dir(os.path.join(output_pth, 'attacks'))

        self.id_landmarks = [i for i in range(config['num_landmarks'])]

    def debug(self, net=None):
        # Print paper figures and debug
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        logic_loss_list = list()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            with torch.no_grad():
                heatmap, regression_y, regression_x = self.model(img)
            pred_landmark = voting(heatmap, regression_y, regression_x, self.Radius)
            logic_loss = BCELoss()
            logic_loss = logic_loss(heatmap, mask)


            regression_x, regression_y = regression_x*mask, regression_y*mask
            to_Image(heatmap[0][0], show="heatmap", normalize=False)
            to_Image(guassian_mask[0][0], show="mask")
            to_Image(regression_x[0][0], show="regression_x", normalize=True)
            to_Image(offset_x[0][0], show="offset_x", normalize=True)
            to_Image(regression_y[0][0], show="regression_y", normalize=True)
            to_Image(offset_y[0][0], show="offset_y", normalize=True)
            image_gt = visualize(img, landmark_list, highlight=25)
            image_pred = visualize(img, pred_landmark, [0, 2])
            image_gt.save('gt.png')
            image_pred.save('pred.png')
            import ipdb; ipdb.set_trace()

            logic_loss_list.append(logic_loss.cpu().item())
            # import ipdb; ipdb.set_trace()
        print(sum(logic_loss_list)/self.dataset.__len__())

    def test(self, net=None):
        self.evaluater.reset()
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        ID = 0

        distance_list = dict()
        mean_list = dict()
        for i in range(19):
            distance_list[i] = list()
            mean_list[i] = list()

        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()

            heatmap, regression_y, regression_x = self.model(img)

            # Vote for the final accurate point
            pred_landmark = voting(heatmap, regression_y, regression_x, self.Radius)

            self.evaluater.record(pred_landmark, landmark_list)
            
            # Optional Save viusal results
            image_pred = visualize(img, pred_landmark)
            image_pred.save(os.path.join(self.output_dir, str(ID)+'_pred.png'))
            image_gt = visualize(img, landmark_list)
            image_pred.save(os.path.join(self.output_dir, str(ID)+'_gt.png'))
            ID += 1
            
        self.evaluater.cal_metrics()

        # For calcuating the minimum distance for each landmark

        #     for i in range(19):
        #         minimum = 9999
        #         mean = 0
        #         temp = list()
        #         for j in range(19):
        #             distance = np.linalg.norm([landmark_list[i][0] - landmark_list[j][0],\
        #                 landmark_list[i][1] - landmark_list[j][1]])
        #             mean += distance
        #             temp.append(distance)
        #             if distance < minimum and distance > 0: minimum = distance
        #         distance_list[i].append(minimum)
        #         temp.sort()
        #         mean_list[i].append(np.array(temp[1:7]).mean())
        # for i in range(19):
        #     distance_list[i] = np.array(distance_list[i]).mean()
        #     mean_list[i] = np.array(mean_list[i]).mean()
        # import pickle
        # with open('distance.pkl', 'wb') as f:
        #     pickle.dump(mean_list, f)
        # with open('top5.pkl', 'wb') as f:
        #     pickle.dump(distance_list, f)
        # import ipdb; ipdb.set_trace()
    
    def attack(self):
        counter = 0
        self.attacker = FGSMAttack(self.model, total_loss, \
            total_loss_adaptive, self.config['lambda'], self)
        
        # # DEBUG
        # attack_box = {  9: [[400, 600], [500, 700]],
        #                 12: [[400, 600], [100, 300]],
        #                 8: [[100, 300], [500, 700]],
        #                 10: [[100, 300], [100, 300]],
        #                 5: [[240, 440], [300, 500]]} #[[100, 600], [250, 750]]
        self.evaluater.reset()
        for epoch in range(2):
            for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(self.dataloader_1):
                img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                    offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()

                # Gen Random Attack Sample
                attack_sample = dict()
                attack_chosen = random.sample(self.id_landmarks, random.randint(1,\
                    self.config['num_landmarks'] - 2))
                # for key, value in attack_box.items():
                #     attack_sample[key] = [random.randint(value[0][0], value[0][1]),\
                #         random.randint(value[1][0], value[1][1])]
                for item in attack_chosen:
                    attack_sample[item] = [random.randint(100, 600),random.randint(250, 750)]
                
                # attack_sample = dict()
                # attack_sample = {0:[400, 600], 2: [300, 400]}
                # attack_sample = get_MICCAI('A')

                # adv_img = self.attacker.FGSM_Target(img, attack_sample, mode=0, debug=False, gt=landmark_list)
                task_list = [1]
                for i in task_list:
                    adv_img = self.attacker.FGSM_Target(img, attack_sample, mode=i, debug=True, gt=landmark_list)
        
                counter += 1
                if counter >= 1:
                    break
            self.evaluater.cal_metrics_attack(task_list)
            return

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='test', help="position of the output dir")
    parser.add_argument("--debug", default='', help="position of the output dir")
    parser.add_argument("--iteration", default='', help="position of the output dir")
    parser.add_argument("--attack", default='', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--checkpoint_file", default="", help="default configs")
    parser.add_argument("--output_file", default="", help="default configs")
    parser.add_argument("--train", default="", help="default configs")
    parser.add_argument("--rand", default="", help="default configs")
    parser.add_argument("--epsilon", default="8", help="default configs")
    args = parser.parse_args()

    with open(os.path.join("runs", args.tag, args.config_file), "r") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
        
    if args.iteration == '':
        iteration = config['last_epoch']
    else:
        iteration = int(args.iteration)
    
    # Load model
    # net = UNet(3, config['num_landmarks']).cuda()
    # net = Runnan(3, config['num_landmarks']).cuda()
    net = UNet_Pretrained(3, config['num_landmarks']).cuda()

    logger.info("Loading checkpoints from epoch {}".format(iteration))
    checkpoints = torch.load(os.path.join(config['runs_dir'], \
                        "model_epoch_{}.pth".format(iteration)))
    net.load_state_dict(checkpoints)
    net = torch.nn.DataParallel(net)

    tester = Tester(logger, config, net, args.tag, args.train, args)
    if args.debug != '':
        tester.debug()

    if args.attack == '':
        tester.test()
    else:
        tester.attack()
