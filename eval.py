import torch
import numpy as np

class Evaluater(object):
    def __init__(self, logger, size, original_size):
        self.pixel_spaceing = 0.1
        self.logger = logger
        self.scale_rate_y = original_size[0] / size[0]
        self.scale_rate_x = original_size[1] / size[1]
        
        self.RE_list = list()

        self.recall_radius = [2, 2.5, 3, 4] # 2mm etc
        self.recall_rate = list()
    
    def reset(self):
        self.RE_list.clear()

    def record(self, pred, landmark):
        # n = batchsize = 1
        # pred : list[ c(y) ; c(x) ]
        # landmark: list [ (x , y) * c]
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float) # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y 
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:,0], 2) + np.power(diff[:,1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        # for i in range(len(Radial_Error)):
        #     if Radial_Error[i] > 10:
        #         print("Landmark {} RE {}".format(i, Radial_Error[i]))
        # if Radial_Error.max() > 10:
        #     return Radial_Error.argmax()
        return None
    
    def cal_metrics(self):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        self.logger.info(Mean_RE_channel)
        self.logger.info("ALL MRE {}".format(Mean_RE_channel.mean()))

        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            self.logger.info("ALL SDR {}mm  {}".format\
                (radius, shot * 100 / total))
