import copy
import numpy as np
from collections import Iterable

import torch
import torch.nn as nn

from utils import to_Image, voting, visualize
from torch.autograd import Variable

class FGSMAttack(object):
    def __init__(self, model, loss_fn, loss_fn_adaptive, lamda, tester):

        self.model = model
        self.epsilon = float(tester.args.epsilon)
        self.loss_fn = loss_fn
        self.loss_fn_adaptive = loss_fn_adaptive
        self.dummy = None
        self.threashold = 0.2
        self.Radius = tester.Radius
        self.lamda = lamda
        self.guassian_mask = tester.dataset.guassian_mask
        self.mask = tester.dataset.mask
        self.offset_x = tester.dataset.offset_x
        self.offset_y = tester.dataset.offset_y
        self.num_iters = 300
        self.alpha = 0.05
        self.evaluater = tester.evaluater
        self.rank_rate = 0.5
        self.scope = 2

        self.scale = 1 / 256 # for nomalization to [-1, 1]
        self.logger = tester.logger
    
    def clip(self, tensor):
        tensor = tensor - (tensor > 1.0) * (tensor - 1.0)
        tensor = tensor - (tensor < -1.0) * (tensor + 1.0)
        return tensor
    
    def gen_attack_gt(self, id_landmark, landmark, mask, heatmap, offset_y, offset_x):
        # landmark: [x, y]
        to_Image(heatmap[0][0], show='test')
        heatmap[0][id_landmark] = 0
        offset_x[0][id_landmark] = 0
        offset_y[0][id_landmark] = 0

        x, y = heatmap[0][0].shape[-1], heatmap[0][0].shape[-2] 
        margin_x_left = max(0, landmark[0] - self.Radius)
        margin_x_right = min(x, landmark[0] + self.Radius)
        margin_y_bottom = max(0, landmark[1] - self.Radius)
        margin_y_top = min(y, landmark[1] + self.Radius)

        mask[0][id_landmark][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        heatmap[0][id_landmark][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.guassian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_x[0][id_landmark][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_y[0][id_landmark][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        
        # import ipdb; ipdb.set_trace()
        # to_Image(heatmap[0][0], show='test')
        # to_Image(offset_x[0][0], show='test', normalize=True)
        # to_Image(offset_y[0][0], show='test', normalize=True)

    def FGSM_Target(self, input, landmark_attack, mode=0, epsilon=None, debug=False, gt=None):
        # Landmark_attack : dict {0: [x, y]}
        split_iter = [1, 20, 50, 99, 150, 200, 250, 299, 400, 600, 750, 999]
        # split_iter = [299]
        if mode in [0, 2]:
            loss_fn = self.loss_fn
        else:
            loss_fn = self.loss_fn_adaptive
        raw = input
        # if gt is not None:
        #     for item in landmark_attack.items():
        #         gt[item[0]] = item[1]

        if debug and gt is not None:
            # Insert attack_landmark to gt
            blue_landmarks = list()
            for item in landmark_attack.items():
                blue_landmarks.append(gt[item[0]] + [item[0]])
                gt[item[0]] = item[1]
            image_gt = visualize(raw, gt, list(landmark_attack.keys()), blue_landmarks)
            image_gt.save(self.evaluater.tag + 'Final_GT.png')
            # import ipdb; ipdb.set_trace()

        if self.epsilon is None:
            self.epsilon = epsilon
        
        with torch.no_grad():
            gt_heatmap, gt_offset_y, gt_offset_x = self.model(input)
        # pred_landmark = voting(gt_heatmap, gt_offset_y, gt_offset_x, self.Radius)
        mask = (gt_heatmap > 0.5).float()

        for item in landmark_attack.items():
            self.gen_attack_gt(item[0], item[1], mask, gt_heatmap, gt_offset_y, gt_offset_x)
        
        total_pertubaton = torch.zeros_like(input).cuda()
        for i in range(self.num_iters):
            input = torch.tensor(input.data, requires_grad=True).cuda()
            heatmap, offset_y, offset_x = self.model(input)

            loss = loss_fn(mask, gt_heatmap, heatmap, gt_offset_y, gt_offset_x, \
                offset_y, offset_x, self.lamda, landmark_attack)
            loss.backward()
            # self.logger.info("Attacking: Iter {} Loss {} eps {}".format(\
            #     i, loss, self.epsilon))
            if i == 0:
                max_loss = loss

            
            grad = input.grad
            # print(grad.view(-1).max() / grad.abs().view(-1).topk(k=768000)[0][-1])
            if mode in [2, 3]:
                # Deprecated
                # scope = self.scope * loss / max_loss
                scope = 1
                threashold = grad.view(-1).max() / 1000.0
                grad_sign = torch.clamp(grad / threashold, -1, 1)
            else:
                grad_sign = grad.sign()
                scope = 1

            pertubation = self.scale * self.epsilon * grad_sign * self.alpha * scope
            total_pertubaton += pertubation
            total_pertubaton = torch.clamp(total_pertubaton.data, -self.scale*self.epsilon,\
                self.scale*self.epsilon)
            input = raw - total_pertubaton
            input = torch.clamp(input.data, -1, 1)

            if i in split_iter:
                adversarial = input.data
                adv_heatmap, adv_offset_y, adv_offset_x = self.model(adversarial)
                loss = loss_fn(mask, gt_heatmap, adv_heatmap, gt_offset_y, gt_offset_x,\
                    adv_offset_y, adv_offset_x, self.lamda, landmark_attack)
                # self.logger.info("Attacking: Iter {} Loss {} eps {}".format(\
                #     i, loss, self.epsilon))
                pred_landmark = voting(adv_heatmap, adv_offset_y, adv_offset_x, self.Radius)
                self.evaluater.record_attack(pred_landmark, gt, list(landmark_attack.keys()), mode, i)

        adversarial = input.data
        

        # if debug:
        #     pertubation = (adversarial - raw)[0] * 8 + 0.5
        #     to_Image(pertubation, show=self.evaluater.tag + 'A_Pertubations')
        #     adv_heatmap, adv_offset_y, adv_offset_x = self.model(adversarial)
        #     # print("Before Target Attack Loss: {}".format(loss))
        #     to_Image(gt_heatmap[0][0], show='A_Mask_Target')
        #     to_Image(adv_heatmap[0][0], show='A_Heatmap')
        #     to_Image(gt_heatmap[0][0], show='B_Mask_Target')
        #     to_Image(adv_heatmap[0][0], show='B_Heatmap')
        #     adv_offset_x = adv_offset_x * mask
        #     to_Image(adv_offset_x[0][0], show='A_Offset_y', normalize=True)
        #     to_Image(gt_offset_x[0][0], show='A_gt_Offset_y', normalize=True)
        #     to_Image(adv_offset_y[0][1], show='B_Offset_y', normalize=True)
        #     to_Image(adv_heatmap[0][1], show='B_Heatmap')
        #     to_Image(adversarial[0], show='A_Adv_Sample', normalize=True)
        #     to_Image(raw[0], show='A_Raw', normalize=True)
        #     pred_landmark = voting(adv_heatmap, adv_offset_y, adv_offset_x, self.Radius)
        #     image_attack = visualize(adversarial, pred_landmark, list(landmark_attack.keys()))
        #     image_attack.save(self.evaluater.tag + 'Final_Attack.png')
        #     self.evaluater.cal_metrics_attack([mode])
        #     import ipdb; ipdb.set_trace()

        return adversarial

    def FGSM_Untarget(self, input, epsilon=None, debug=True):
        if self.epsilon is None:
            self.epsilon = epsilon
        
        with torch.no_grad():
            gt_heatmap, gt_offset_y, gt_offset_x = self.model(input)
        # pred_landmark = voting(gt_heatmap, gt_offset_y, gt_offset_x, self.Radius)
        mask = (gt_heatmap > 0.5).float()

        input = torch.tensor(input.data, requires_grad=True).cuda()
        heatmap, offset_y, offset_x = self.model(input)

        bkp_loss = self.loss_fn(mask, heatmap, gt_offset_y, gt_offset_x, \
            offset_y, offset_x, self.lamda)
        bkp_loss.backward()
        grad = input.grad
        grad_sign = grad

        pertubation = self.epsilon * grad_sign * self.scale
        adversarial = input - pertubation
        adversarial = self.clip(adversarial)


        if debug:
            print("Before Target Attack Loss: {}".format(bkp_loss))
            for e in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
                pertubation = self.scale * e * grad_sign
                adversarial = input + pertubation
                adversarial = self.clip(adversarial)
                adv_heatmap, adv_offset_y, adv_offset_x = self.model(adversarial)
                loss = self.loss_fn(mask, adv_heatmap, gt_offset_y, gt_offset_x,\
                    adv_offset_y, adv_offset_x, self.lamda)
                print("After Target Attack Loss: {} epsilon: {}".format(loss, e))
                to_Image(mask[0][0], show='A_Mask_Target')
                to_Image(adv_heatmap[0][0], show='A_Heatmap')
                to_Image(adversarial[0], show='A_Adv_Sample', normalize=True)
                to_Image(input[0], show='A_Raw', normalize=True)
            import ipdb; ipdb.set_trace()

from utils import to_Image
from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

# class FGSMAttack(object):
#     def __init__(self, model, loss_fn, epsilon=None):

#         self.model = model
#         self.epsilon = epsilon
#         self.loss_fn = loss_fn
#         self.dummy = None
#         self.threashold = 0.2

#         self.scale = 1 / 128 # for nomalization to [-1, 1]

#     def FGSM_Untarget_Heatmap(self, input, epsilons=None, debug=True):
#         """
#         Given examples (X_nat, y), returns their adversarial
#         counterparts with an attack length of eps
#         ilon.
#         """
#         # Providing epsilons in batch
#         if epsilons is not None:
#             self.epsilon = epsilons      

#         if debug:
#             to_Image(input[0], show="A_Raw", normalize=True)
        
#         with torch.no_grad():
#             bkp_heatmap, _, __ = self.model(input) 
#         mask = (bkp_heatmap > 0.5).float()

#         input = to_var(input, requires_grad=True)

#         heatmap, regression_y, regression_x = self.model(input) 
#         if self.dummy is None:
#             self.dummy = torch.zeros(heatmap.shape, dtype=torch.float).cuda()
        
#         loss = self.loss_fn(heatmap, mask)
#         loss.backward()

#         grad_sign = input.grad.sign()
#         pertubation = self.scale * self.epsilon * grad_sign
#         adversarial = pertubation + input
#         # import ipdb; ipdb.set_trace()
#         # adversarial = np.clip(adversarial, -1, 1)

#         adv_heatmap, _, __ = self.model(adversarial)

#         if debug:
#             print("Before Attack Loss: {}".format(loss))
#             loss = self.loss_fn(adv_heatmap, mask)
#             print("After Attack Loss: {}".format(loss))
#             to_Image(mask[0][0], show='A_Mask_Untarget')
#             to_Image(heatmap[0][0], show='A_heatmap')
#             to_Image(adv_heatmap[0][0], show='A_Dummy')
#             to_Image(adv_heatmap[0] - input[0], show='A_Pertubation')
#             to_Image(input[0], show='A_sample', normalize=True)
#             import ipdb; ipdb.set_trace()
            
#         return adversarial
