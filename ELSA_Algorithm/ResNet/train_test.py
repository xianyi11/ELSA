import os
import time
import torch
from tqdm import tqdm
import misc
import math
from typing import Iterable, Optional
import lr_sched
import sys
from timm.utils import accuracy
from models import l2_regularization, get_absoluteValue
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# device = "cpu"

def quan_test(model, data_loader, verbose=False):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.to(device)
    model.eval()
    correct = 0
#     print('----- Model Evaluation -----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(data_loader, total=len(data_loader.batch_sampler), desc="Testing"):
            # Forward pass.
            # features = torch.clamp(torch.round(features*255),0,255)
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    if verbose:
        print(f'Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)')
    return percent


def test(model, data_loader, verbose=False):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.to(device)
    model.eval()
    correct = 0
#     print('----- Model Evaluation -----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(data_loader, total=len(data_loader.batch_sampler), desc="Testing"):
            # Forward pass.
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    if verbose:
        print(f'Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)')
    return percent

def get_logits_loss(fc_t, fc_s, one_hot_label, temp, num_classes=1000):
    s_input_for_softmax = fc_s / temp
    t_input_for_softmax = fc_t / temp

    softmax = torch.nn.Softmax(dim=1)
    logsoftmax = torch.nn.LogSoftmax()

    t_soft_label = softmax(t_input_for_softmax)

    softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

    fc_s_auto = fc_s.detach()
    fc_t_auto = fc_t.detach()
    log_softmax_s = logsoftmax(fc_s_auto)
    log_softmax_t = logsoftmax(fc_t_auto)
    # one_hot_label = F.one_hot(label, num_classes=num_classes).float()
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss

    soft_loss = (temp ** 2) * torch.mean(softmax_loss)

    return soft_loss

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        # print("image",images[0,0,0,:])
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        if hasattr(model, "reset"):
            model.reset()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pertimestep(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        # print("image",images[0,0,0,:])
        output,accu_list = model(images,True)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_pertimestep = []
        for accu in accu_list:
            acc1, acc5 = accuracy(accu, target, topk=(1, 5))
            acc1_pertimestep.append(acc1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        for t,acc1 in enumerate(acc1_pertimestep):
            metric_logger.meters[f'acc1_{t}'].update(acc1.item(), n=batch_size)
        
        if hasattr(model, "reset"):
            model.reset()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    str = "* "
    for t in range(len(acc1_pertimestep)):
        str = str + "Acc_{T}:{top1.global_avg:.3f} ".format(T=t,top1=eval(f"metric_logger.acc1_{t}"))
    
    print(str)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


torch.no_grad()
def init_quan(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn=None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
        
        
        loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss/accum_iter
        loss.backward()
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=False,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def set_quan_scale(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn=None, log_writer=None,
                    args=None, num_classes=1000):
    model.eval()
    # print("===============set_quan_scale==============")
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        break

def train_one_epoch_distill(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn=None, log_writer=None,
                    args=None, num_classes=1000):
    model.train(True)
    model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            targets_for_onehot = targets + 0
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            with torch.no_grad():
                outputs_teacher = model_teacher(samples)
        loss = criterion(outputs, targets)
        if mixup_fn is not None:
            targets_one_hot = torch.nn.functional.one_hot(targets_for_onehot, num_classes=num_classes)
        else:
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        soft_loss = get_logits_loss(outputs_teacher, outputs, targets_one_hot, 2.0)
        loss = loss + soft_loss

        if args.actPrune:
            absoluteValue = get_absoluteValue(model)/args.actPruneScale
            l2_loss = l2_regularization(model)/args.actPruneScale
            loss = loss + l2_loss*args.actPruneRatio
            l2_loss_value = l2_loss.item()

        loss_value = loss.item()
        soft_loss_value = soft_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss/accum_iter
        loss.backward()
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=False,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(soft_loss=soft_loss_value)
        if args.actPrune:
            metric_logger.update(act_l2_loss=l2_loss_value)
            metric_logger.update(act_absoluteValue=absoluteValue)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train(model, criterion, data_loader, test_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model.""" 
    
    # Move model to the device (CPU or GPU).
    model.to(device)
    metric_logger = misc.MetricLogger(delimiter="  ")

    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

#     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        tick = time.time()
        model.train()
        # Loop over data.
        for batch_idx, (features, target) in tqdm(enumerate(data_loader), total=len(data_loader.batch_sampler), desc="training"):
            # Forward pass.
            output = model(features.to(device))
            loss = criterion(output.to(device), target.to(device))
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01
            tock = time.time()
        acc = test(model, test_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print('Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds'.format(epoch+1, ema_loss, tock-tick),)
        torch.save(model.state_dict(), f'model_{epoch}.ckpt')
        print("Model Saved!")
        if os.path.isfile(f'model_{epoch-1}.ckpt'):
            os.remove(f'model_{epoch-1}.ckpt')
    return accs

