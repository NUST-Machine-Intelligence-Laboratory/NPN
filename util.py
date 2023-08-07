import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import shutil
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.backends.cudnn as cudnn
import numpy as np
from json import dump
import random
import math
import kornia
import kornia.augmentation as kaug
from kornia.augmentation.container import AugmentationSequential



default_float_dtype = torch.get_default_dtype()
def min_max(x):
    return (x - x.min())/(x.max() - x.min())
# misc
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = True
    torch.cuda.empty_cache()

class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w, x_s
def set_device(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        print(f'Available GPUs Index : {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except KeyError:
        print('No GPU available, using CPU ... ')
    return torch.device('cuda') if torch.cuda.device_count() >= 1 else torch.device('cpu')


def save_params(params, params_file, json_format=False):
    with open(params_file, 'w') as f:
        if not json_format:
            params_file.replace('.json', '.txt')
            for k, v in params.__dict__.items():
                f.write(f'{k:<20}: {v}\n')
        else:
            params_file.replace('.txt', '.json')
            dump(params.__dict__, f, indent=4)


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


# model-related
def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias.data, val=0)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, val=0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def freeze_layer(module):
    for parameters in module.parameters():
        parameters.requires_grad = False


def unfreeze_layer(module):
    for parameters in module.parameters():
        parameters.requires_grad = True


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_scale_factor, projection_size, init_method='He', activation='relu', use_bn=True):
        super().__init__()

        mlp_hidden_size = round(mlp_scale_factor * in_channels)
        if activation == 'relu':
            non_linear_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky relu':
            non_linear_layer = nn.LeakyReLU(inplace=True)
        elif activation == 'tanh':
            non_linear_layer = nn.Tanh()
        else:
            raise AssertionError(f'{activation} is not supported yet.')
        mlp_head_module_list = [nn.Linear(in_channels, mlp_hidden_size)]
        if use_bn: mlp_head_module_list.append(nn.BatchNorm1d(mlp_hidden_size))
        mlp_head_module_list.append(non_linear_layer)
        mlp_head_module_list.append(nn.Linear(mlp_hidden_size, projection_size))

        self.mlp_head = nn.Sequential(*mlp_head_module_list)
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)


# optimizer, scheduler 
def build_sgd_optimizer(params, lr, weight_decay, nesterov=True):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)


def build_adam_optimizer(params, lr, weight_decay=0, amsgrad=False):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=amsgrad)


def build_lr_plan(lr, total_epochs, warmup_epochs, warmup_lr=0.1, decay='linear', warmup_gradual=False):
    def make_linear_lr(init_lr, last, end_lr):
        return list(np.linspace(init_lr, end_lr, last))

    def make_cosine_lr(init_lr, last, end_lr=0.0):
        assert end_lr < init_lr
        lrs = [init_lr] * last
        for i in range(last):
            lrs[i] = (init_lr - end_lr) * 0.5 * (1 + math.cos(i / last * math.pi)) + end_lr
        return lrs

    if warmup_gradual:
        warmup_lr_plan = make_linear_lr(warmup_lr * 1e-5, warmup_epochs, warmup_lr)
    else:
        warmup_lr_plan = [warmup_lr] * warmup_epochs
    if decay == 'baseline':
        lr_plan = [lr] * total_epochs
        for ep in range(total_epochs):
            lr_plan[ep] = lr * (0.1 ** int(ep >= 60)) * (0.1 ** int(ep >= 120)) * (0.1 ** int(ep >= 150))
        return lr_plan
    elif decay == 'step':
        lr_plan = [lr] * total_epochs
        step = [60, 120, 150]
        for ep in range(total_epochs):
            lr_plan[ep] = lr * (0.1 ** int(ep >= 60)) * (0.1 ** int(ep >= 120)) * (0.1 ** int(ep >= 150))
        lr_plan[:warmup_epochs] = warmup_lr_plan
        return lr_plan
    elif decay == 'step30':
        lr_plan = [lr] * total_epochs
        step = [50, 80, 100]
        for ep in range(total_epochs):
            lr_plan[ep] = lr * (0.1 ** int(ep >= step[0])) * (0.1 ** int(ep >= step[1])) * (0.1 ** int(ep >= step[2]))
        lr_plan[:warmup_epochs] = warmup_lr_plan
        return lr_plan
    elif decay == 'linear':
        lr_plan = warmup_lr_plan
        lr_decay_start = 60
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_linear_lr(lr, total_epochs - lr_decay_start, lr * 0.00001)
        return lr_plan
    elif decay == 'cosine':
        lr_plan = warmup_lr_plan
        lr_decay_start = 60
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_cosine_lr(lr, total_epochs - lr_decay_start)
        return lr_plan
    elif decay == 'cosineRestart':
        lr_plan = warmup_lr_plan
        step = [60, 90, 120]
        lr_plan += [lr] * (step[0] - warmup_epochs)
        lr_plan += make_cosine_lr(lr, step[1] - step[0])
        # lr *= 2
        lr_plan += [lr] * (step[2] - step[1])
        lr_plan += make_cosine_lr(lr, total_epochs - step[2])
        return lr_plan
    elif decay == 'cosineNew':
        lr_plan = warmup_lr_plan
        step = [60, 90]
        lr_plan += [lr] * (step[0] - warmup_epochs)
        cos_lr = make_cosine_lr(lr, step[1] - step[0])
        lr_plan += cos_lr[:-3]
        lr_plan += make_cosine_lr(cos_lr[-3], total_epochs - step[1] + 3)
        return lr_plan
    elif decay.startswith('step:'):
        lr_plan = [lr] * total_epochs
        ele = decay.split(':')[1].split(',')
        step = [int(i) for i in ele[:-1]]
        last_ele = float(ele[-1])
        if last_ele >= 1.0:
            step.append(int(last_ele))
            step_factor = 0.1
        else:
            step_factor = last_ele
        for ep in range(total_epochs):
            decay_factor = 1.0
            for i in range(len(step)):
                decay_factor *= (step_factor ** int(ep >= step[i]))
            lr_plan[ep] = lr * decay_factor
        lr_plan[:warmup_epochs] = warmup_lr_plan
        return lr_plan
    elif decay.startswith('linear:'):
        lr_plan = warmup_lr_plan
        ele = decay.split(':')[1].split(',')
        lr_decay_start = int(ele[0])
        end_lr = float(ele[1]) if len(ele) == 2 else lr * 0.00001
        lr_decay_end = int(ele[2]) if len(ele) == 3 and int(ele[2]) > lr_decay_start else total_epochs
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_linear_lr(lr, lr_decay_end - lr_decay_start, end_lr)
        lr_last = lr_plan[-1]
        lr_plan += [lr_last] * (total_epochs - lr_decay_end)
        return lr_plan
    elif decay.startswith('cosine:'):
        lr_plan = warmup_lr_plan
        ele = decay.split(':')[1].split(',')
        lr_decay_start = int(ele[0])
        end_lr = float(ele[1]) if len(ele) >= 2 else 0.0
        lr_decay_end = int(ele[2]) if len(ele) == 3 and int(ele[2]) > lr_decay_start else total_epochs
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_cosine_lr(lr, lr_decay_end - lr_decay_start, end_lr)
        lr_last = lr_plan[-1]
        lr_plan += [lr_last] * (total_epochs - lr_decay_end)
        return lr_plan
    elif decay.startswith('halfcosine:'):
        lr_plan = warmup_lr_plan
        ele = decay.split(':')[1].split(',')
        lr_decay_start = int(ele[0])
        end_lr = float(ele[1]) if len(ele) >= 2 else 0.0
        lr_decay_end = int(ele[2]) if len(ele) == 3 and int(ele[2]) > lr_decay_start else total_epochs
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_cosine_lr(lr, 2 * (lr_decay_end - lr_decay_start), end_lr)[lr_decay_end - lr_decay_start:]
        lr_last = lr_plan[-1]
        lr_plan += [lr_last] * (total_epochs - lr_decay_end)
        return lr_plan
    else:
        raise AssertionError(f'lr decay method: {decay} is not implemented yet.')


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# others
def kl_div(p, q, base=2):
    # p, q is in shape (batch_size, n_classes)
    if base == 2:
        return (p * p.log2() - p * q.log2()).sum(dim=1)
    else:
        return (p * p.log() - p * q.log()).sum(dim=1)


def symmetric_kl_div(p, q, base=2):
    return kl_div(p, q, base) + kl_div(q, p, base)


def js_div(p, q, base=2):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, base) + 0.5 * kl_div(q, m, base)


def entropy(p):
    return Categorical(probs=p).entropy()


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def check_nan(tensor):
    return torch.isnan(tensor).any()


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def linear_rampup2(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


class ReAug(object):
    def __init__(self, strong_aug='auto', dataset='cifar10'):
        self.transform_weak = None
        self.transform_strong = None
        self.init_img_augment(strong_aug, dataset)

    def init_img_augment(self, strong_aug, dataset):
        # cifar
        if dataset in ['cifar10', 'cifar100']:
            mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            std  = [x / 255.0 for x in [ 63.0,  62.1,  66.7]]
            img_size = 32
        elif dataset in ['Clothing1M']:
            mean = [0.6959, 0.6537, 0.6371]
            std = [0.3113, 0.3192, 0.3214]
            img_size = 224
        else:
            raise AssertionError(f'ReAug for {dataset} is not implemented yet!')
        if dataset in ['cifar10', 'cifar100']:
            rand_crop = kaug.RandomCrop(size=(img_size, img_size), padding=(2, 2, 2, 2), padding_mode='reflect')
            print("dataset in ReAug is " + dataset)
        elif dataset in ['Clothing1M']:
            rand_crop = kaug.RandomResizedCrop(size=(img_size, img_size), scale=(1.05, 1.25))
        else:
            raise AssertionError(f'ReAug for {dataset} is not implemented yet!')

        self.transform_weak = AugmentationSequential(
            # kaug.Denormalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            rand_crop,
            kaug.RandomHorizontalFlip(),
            # kaug.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            same_on_batch=False,
        )

        self.transform_strong = AugmentationSequential(
            kaug.Denormalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            IntermediateTransform(strong_aug),
            kaug.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            same_on_batch=False,
        )

    def __call__(self, sample, mode='w'):
        if mode == 'w':
            return self.transform_weak(sample)
        elif mode == 's':
            return self.transform_strong(sample)
        else:
            raise AssertionError(f'ReAug mode {mode} is not supported!')


class IntermediateTransform(nn.Module):
    def __init__(self, aug_type='auto'):
        super().__init__()
        transforms_list = [
            torchvision.transforms.Lambda(lambda x: (x*255.0).type(torch.uint8)),
            None,
            torchvision.transforms.Lambda(lambda x: x.to(dtype=default_float_dtype).div(255))
        ]
        if aug_type.startswith('rand'):
            n_ops = 10 if '-' not in aug_type else int(aug_type.split('-')[1])
            transforms_list[1] = torchvision.transforms.RandAugment(num_ops=n_ops, magnitude=10)
        elif aug_type in ['auto', 'auto-cifar']:
            transforms_list[1] = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        elif aug_type == 'auto-imagenet':
            transforms_list[1] = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET)
        else:
            raise AssertionError(f'Augment Type: {aug_type} is not supported.')
        self.transform_tmp = torchvision.transforms.Compose(transforms_list)
        # self.transform_tmp = torchvision.transforms.Compose([
        #     torchvision.transforms.Lambda(lambda x: (x*255.0).type(torch.uint8)),
        #     torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10) if aug_type == 'auto' else torchvision.transforms.RandAugment(num_ops=n_ops, magnitude=10),
        #     torchvision.transforms.Lambda(lambda x: x.to(dtype=default_float_dtype).div(255)),
        # ])

    def forward(self, x):
        return self.transform_tmp(x)


class EMA(object):
    """
    Usage:
        model = ResNet(config)
        ema = EMA(model, alpha=0.999)
        ... # train an epoch
        ema.update_params(model)
        ema.apply_shadow(model)
    """
    def __init__(self, model, alpha=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]
        self.alpha = alpha

    def init_params(self, model):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]

    def update_params(self, model):
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(self.alpha * self.shadow[name] + (1 - self.alpha) * state[name])

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def set_update_step(self, new_alpha):
        self.alpha = new_alpha


class Queue(object):
    def __init__(self, n_samples, n_classes, memory_length=5):
        super().__init__()
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.memory_length = memory_length
        # the item in content_dict is as follows:
        #   dict {
        #         key: 'pred', value: [(pred class index, pred probability), ...];
        #         key: 'loss', value: [ loss, ... ];
        #         key: 'most_prob_label', value: predicted label with highest accumulated probability
        #        }
        self.content = np.array([
            {'pred': [], 'loss': [], 'label': -1, 'pred_dist': []} for i in range(n_samples)
        ])
        self.most_prob_labels = torch.Tensor([-1 for i in range(n_samples)]).long() # TODO: store distributuion instead of label
        self.accumulated_pred = torch.zeros(n_samples, n_classes).float()

    def update(self, indices, losses, scores, labels):
        probs, preds = scores.max(dim=1)
        for i in range(indices.shape[0]):
            if len(self.content[indices[i].item()]['pred']) >= self.memory_length:
                self.content[indices[i].item()]['pred'].pop(0)
                self.content[indices[i].item()]['pred_dist'].pop(0)
                if losses is not None:
                    self.content[indices[i].item()]['loss'].pop(0)
            self.content[indices[i].item()]['pred'].append((preds[i].item(), probs[i].item()))
            self.content[indices[i].item()]['pred_dist'].append(scores[i].cpu())

            if losses is not None:
                try:
                    self.content[indices[i].item()]['loss'].append(losses[i].item())
                except:
                    print(indices.shape, losses.shape)
                    raise AssertionError()
            if labels is not None:
                self.content[indices[i].item()]['label'] = labels[i].item()

        for i in range(indices.shape[0]):
            tmp = {}
            most_prob_label = -1
            highest_prob = 0
            for pred_idx, pred_prob in self.content[indices[i].item()]['pred']:
                if pred_idx not in tmp:
                    tmp[pred_idx] = pred_prob
                else:
                    tmp[pred_idx] += pred_prob
                if highest_prob < tmp[pred_idx]:
                    highest_prob = tmp[pred_idx]
                    most_prob_label = pred_idx
            self.most_prob_labels[indices[i].item()] = most_prob_label

            accumulated_pred = torch.zeros(self.n_classes)
            for pred_dist in self.content[indices[i].item()]['pred_dist']:
                accumulated_pred += pred_dist
            try:
                self.accumulated_pred[indices[i].item(), :] = accumulated_pred.softmax(dim=0)
            except:
                print(indices[i], self.accumulated_pred.shape)
                print(self.accumulated_pred[indices[i].item(), :].shape)
                print(accumulated_pred.shape)
                raise AssertionError()
