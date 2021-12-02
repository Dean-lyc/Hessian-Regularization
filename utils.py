import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image
    return _cutout

def to_tensor():
    def _to_tensor(image):
        if len(image.shape) == 3:
            return torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))
    return _to_tensor

def normalize(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    def _normalize(image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - mean) / std
        return image

    return _normalize

def get_loader(args):
    if args.dataset_name == "CIFAR10":
        if args.mask_size == 0:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
                cutout(args.mask_size, args.p_cutout, True),
                to_tensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
        trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testLoader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.dataset_name == "CIFAR100":
        if args.mask_size == 0:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]) # meanstd transformation
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
                cutout(8, 1, True),
                to_tensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
        testLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)
    return trainLoader, testLoader




def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class ConfidencePenalty(nn.Module):
    def __init__(self):
        super(ConfidencePenalty, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

class LabelSmoothing(nn.Module):
    def __init__(self):
        super(LabelSmoothing, self).__init__()

    def forward(self, x, classes_num=10):
        b = F.log_softmax(x, dim=1)
        b = -1.0 * b.sum() / classes_num
        return b