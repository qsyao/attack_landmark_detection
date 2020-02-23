import copy
import numpy as np
from collections import Iterable

import torch
import torch.nn as nn

from utils import to_Image
from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

class FGSMAttack(object):
    def __init__(self, model, loss_fn, epsilon=None):

        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.dummy = None
        self.threashold = 0.2

        self.scale = 1 / 128 # for nomalization to [-1, 1]

    def FGSM_Untarget_Heatmap(self, input, epsilons=None, debug=True):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of eps
        ilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons      

        if debug:
            to_Image(input[0], show="A_Raw", normalize=True)
        
        with torch.no_grad():
            bkp_heatmap, _, __ = self.model(input) 
        mask = (bkp_heatmap > 0.5).float()

        input = to_var(input, requires_grad=True)

        heatmap, regression_y, regression_x = self.model(input) 
        if self.dummy is None:
            self.dummy = torch.zeros(heatmap.shape, dtype=torch.float).cuda()
        
        loss = self.loss_fn(heatmap, mask)
        loss.backward()

        grad_sign = input.grad.sign()
        pertubation = self.scale * self.epsilon * grad_sign
        adversarial = pertubation + input
        # import ipdb; ipdb.set_trace()
        # adversarial = np.clip(adversarial, -1, 1)

        adv_heatmap, _, __ = self.model(adversarial)

        if debug:
            print("Before Attack Loss: {}".format(loss))
            loss = self.loss_fn(adv_heatmap, mask)
            print("After Attack Loss: {}".format(loss))
            to_Image(mask[0][0], show='A_Mask_Untarget')
            to_Image(heatmap[0][0], show='A_heatmap')
            to_Image(adv_heatmap[0][0], show='A_Dummy')
            to_Image(adv_heatmap[0] - input[0], show='A_Pertubation')
            to_Image(input[0], show='A_sample', normalize=True)
            import ipdb; ipdb.set_trace()

        return adversarial
