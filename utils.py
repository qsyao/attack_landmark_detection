import numpy as np 
import torch
import time

from multiprocessing import Process, Queue
from PIL import Image
from torchvision.transforms import ToPILImage

to_PIL = ToPILImage()

def distance(pred, landmark, k):
    diff = np.zeros([2], dtype=float) # y, x
    diff[0] = abs(pred[0] - landmark[k][1]) * 3.0
    diff[1] = abs(pred[1] - landmark[k][0]) * 3.0
    Radial_Error = np.sqrt(np.power(diff[0], 2) + np.power(diff[1], 2))
    Radial_Error *= 0.1
    if Radial_Error > 10:
        return Radial_Error
    return 0

def to_Image(tensor, show=None, normalize=False):
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.cpu()
    image = to_PIL(tensor)
    if show:
        image.save(show + ".png")
    return image

def voting_channel(k, heatmap, regression_y, regression_x,\
     Radius, spots_y, spots_x, queue, gt, mask):
    n, c, h, w = heatmap.shape

    score_map = np.zeros([h, w], dtype=int)
    for i in range(int(3.14 * Radius * Radius)):
        vote_x = regression_x[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_y = regression_y[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_x = spots_x[0, k, i] + int(vote_x * Radius)
        vote_y = spots_y[0, k, i] + int(vote_y * Radius)
        if vote_x < 0 or vote_x >= w or vote_y < 0 or vote_y >= h:
            # Outbounds
            continue
        score_map[vote_y, vote_x] += 1
    score_map = score_map.reshape(-1)
    candidataces = score_map.argsort()[-10:]
    candidataces_x = candidataces % w
    candidataces_y = candidataces / w
    # import ipdb; ipdb.set_trace()
    gg = distance([candidataces_y[-1], candidataces_x[-1]], gt, k)
    if gg:
        print("Landmark {} RE {}".format(k, gg))
        print(candidataces_y.astype(int))
        print(candidataces_x.astype(int))
        print(gt[k][1], gt[k][0])
    queue.put([k, score_map.argmax()])

def voting(heatmap, regression_y, regression_x, Radius, gt, mask):
    # n = batchsize = 1
    heatmap = heatmap.cpu()
    regression_x, regression_y = regression_x.cpu(), regression_y.cpu()
    n, c, h, w = heatmap.shape
    assert(n == 1)
    score_map = torch.zeros(n, c, h, w, dtype=torch.int16)
    spots_heat, spots = heatmap.view(n, c, -1).topk(dim=-1, \
        k=int(3.14 * Radius * Radius))
    spots_y = spots / w
    spots_x = spots % w

    # for mutiprocessing debug
    # voting_channel(0, heatmap,\
    #         regression_y, regression_x, Radius, spots_y, spots_x, None, None, None)
            
    # MutiProcessing
    process_list = list()
    queue = Queue()
    for k in range(c):
        process = Process(target=voting_channel, args=(k, heatmap,\
            regression_y, regression_x, Radius, spots_y, spots_x, queue, gt, mask))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()
    
    landmark = np.zeros([c], dtype=int)
    for i in range(c):
        out = queue.get()
        landmark[out[0]] = out[1]

        # This is for guassian mask
        # landmark[i] = heatmap[0][i].view(-1).max(0)[1]
    landmark_y = landmark / w
    landmark_x = landmark % w
    return [landmark_y, landmark_x]
