import argparse

from models import *
from vgg import vgg16_bn
from train_test import evaluate
from utils import *
from fusion import fuse_module_train
import os
import misc
from datasets import build_cifar10_dataset, build_cifar100_dataset
from copy import deepcopy
from snnConverter import *
import glo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--model', default=128, type=str, help='Batch size to use for train/test sets')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size to use for train/test sets')
    parser.add_argument('--dataPath', default="urbansound8k", type=str, help="Relative path of the dataset")
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--quanmodel', default="", type=str, help='Random seed to use')
    parser.add_argument('--WeightBit', default=8, type=int, help='Random seed to use')
    parser.add_argument('--ActBit', default=8, type=int, help='Random seed to use')
    parser.add_argument('--maxTimeStep', default=32, type=int, help='Random seed to use')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dataset', default="imagenet", type=str, help='dataset name')
    args = parser.parse_args()

    if args.model != "vgg16" or args.dataset not in ("cifar10", "cifar100"):
        raise ValueError("This tree only supports --model vgg16 with --dataset cifar10 or cifar100.")

    misc.init_distributed_mode(args)
    device = torch.device("cuda")

    if args.dataset == "cifar10":
        model = vgg16_bn(num_classes=10)
    else:
        model = vgg16_bn(num_classes=100)

    model_teacher = deepcopy(model)
    quantized_train_model_fusebn(model, weightBit=args.WeightBit, actBit=args.ActBit)
    fuse_module_train(model)
    model.classifier[0] = QuanLinear(m=model.classifier[0].m, quan_w_fn=LsqQuan(bit=args.WeightBit, all_positive=False, symmetric=False, per_channel=False),
                                        quan_out_fn=LsqQuan(bit=args.ActBit, all_positive=True, symmetric=False, per_channel=False))
    model.classifier[3] = QuanLinear(m=model.classifier[3].m, quan_w_fn=LsqQuan(bit=args.WeightBit, all_positive=False, symmetric=False, per_channel=False),
                                        quan_out_fn=LsqQuan(bit=args.ActBit, all_positive=True, symmetric=False, per_channel=False))

    batch_size = args.batch_size

    if args.dataset == "cifar10":
        dataset_val = build_cifar10_dataset(is_train=False, args=args)
    else:
        dataset_val = build_cifar100_dataset(is_train=False, args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )

    print(args.gpu)
    model.to(device)
    model_teacher.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu], find_unused_parameters=True)
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))

    model.to(device)
    model.train()

    print(model)

    state_dict = torch.load(args.quanmodel, map_location=device)
    info = model.module.load_state_dict(state_dict, strict=False)
    print(info)
    f = open("QAT_model.txt", "w+")
    f.write(str(model))
    f.close()

    set_init_false(model)
    evaluate(data_loader_val, model, device, args)
    quantized_inference_model_fusebn(model)
    f = open("Inference_model.txt", "w+")
    f.write(str(model))
    f.close()

    glo._init()

    output_bin_qann_dir = f"/data1/user/output_bin_qann_{args.model}_{args.dataset}_w{args.WeightBit}_a{args.ActBit}/"

    if args.rank == 0:
        if not os.path.exists(output_bin_qann_dir):
            os.mkdir(output_bin_qann_dir)
        glo.set_value("output_bin_qann_dir", output_bin_qann_dir)
        save_for_bin(model, output_bin_qann_dir)

    evaluate(data_loader_val, model, device, args)

    model = IntegerSNNWarapper(ANNModel=model, bit=args.ActBit, max_timestep=args.maxTimeStep, modelName=args.model)
    f = open("SNN_model.txt", "w+")
    f.write(str(model))
    f.close()
    calOrder = []
    set_snn_save_name(model, calOrder)

    output_bin_snn_dir = f"/data1/user/output_bin_snn_{args.model}_{args.dataset}_w{args.WeightBit}_a{args.ActBit}_T{args.maxTimeStep}/"
    if args.rank == 0:
        if not os.path.exists(output_bin_snn_dir):
            os.mkdir(output_bin_snn_dir)
        glo.set_value("output_bin_snn_dir", output_bin_snn_dir)

    if args.rank == 0:
        save_for_bin_snn(model, output_bin_snn_dir)

    if args.rank == 0:
        f = open(f"{output_bin_snn_dir}/calculationOrder.txt", "w+")
        for order in calOrder:
            f.write(order + "\n")
        f.close()

    torch.distributed.barrier()
    evaluate(data_loader_val, model, device, args)
