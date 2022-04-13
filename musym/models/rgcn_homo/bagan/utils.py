from easydict import EasyDict as edict
import yaml
import numpy as np
import torch
import os


def conditional_latent_generator(distribution, class_num, batch):
	class_labels = torch.randint(0, class_num, (batch,), dtype=torch.long)
	fake_z = distribution[class_labels[0].item()].sample((1, ))
	for c in class_labels[1:]:
		fake_z = torch.cat((fake_z, distribution[c.item()].sample((1,))), dim=0)
	return fake_z, class_labels


def batch2one(Z, y, z, class_num):
	for i in range(class_num):
        # Z[label][0] should be deleted..
		Z[i] = torch.cat((Z[i], z[y==i].cpu()), dim=0)
	return Z			
	
class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def one_hot(x, num_classes):
        '''
        One-hot encoding of the vector of classes. It uses number of classes + 1 to
        encode fake images
        :param x: vector of output classes to one-hot encode
        :return: one-hot encoded version of the input vector
        '''
        label_numpy = x.data.cpu().numpy()
        label_onehot = np.zeros((label_numpy.shape[0], num_classes + 1))
        label_onehot[np.arange(label_numpy.shape[0]), label_numpy] = 1
        return torch.FloatTensor(label_onehot)


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

#Clear
def save_checkpoint(state, filename='checkpoint'):
    pardir = os.path.abspath(os.path.join(filename, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    torch.save(state, filename + '.pth.tar')

def print_gan_log(epoch, epoches, iteration, iters, learning_rate,
              display, batch_time, data_time, D_losses, G_losses):
    print('epoch: [{}/{}] iteration: [{}/{}]\t'
          'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
    print('Time {batch_time.sum:.3f}s / {0}iters, ({batch_time.avg:.3f})\t'
          'Data load {data_time.sum:.3f}s / {0}iters, ({data_time.avg:3f})\n'
          'Loss_D = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})\n'
          'Loss_G = {loss_G.val:.8f} (ave = {loss_G.avg:.8f})\n'.format(
              display, batch_time=batch_time,
              data_time=data_time, loss_D=D_losses, loss_G=G_losses))

#Clear
def print_vae_log(epoch, epoches, iteration, iters, learning_rate,
              display, batch_time, data_time, losses):

    print('epoch: [{}/{}] iteration: [{}/{}]\t'
          'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
    print('Time {batch_time.sum:.3f}s / {0}iters, ({batch_time.avg:.3f})\t'
          'Data load {data_time.sum:.3f}s / {0}iters, ({data_time.avg:3f})\n'
          'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
              display, batch_time=batch_time,
              data_time=data_time, loss=losses))

