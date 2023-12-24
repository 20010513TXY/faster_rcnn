import random

from torchvision.transforms import functional as F
class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, image,target):
        for t in self.transforms:
            image,target = t(image,target) #依次应用transforms中的每个操作
        return image,target

class ToTensor(object):
    def __call__(self, image,target):
        image = F.to_tensor(image)
        return image,target


class RandomHorizontalFlip(object):
    def __init__(self,prob = 0.5):
        self.prob = prob

    def __call__(self, image,target):
        if random.random() < self.prob:
            height,width = image.shape[-2:]
            # print('图片水平翻转前的宽={}、高={}'.format(height,width))
            image = image.flip(-1) #图片水平翻转
            # 真实边界框翻转
            bbox = target['boxes']
            bbox[:,[0,2]] = width - bbox[:,[2,0]]
            # print('bbox={}'.format(bbox)) #bbox=tensor([[  0.,   1., 428., 375.]])
            # print('bbox[:,[2,0]]={},bbox[:,[0,2]]={}'.format(bbox[:,[2,0]],bbox[:,[0,2]]))
            target['boxes'] = bbox
        return image,target
