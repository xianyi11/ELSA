import argparse

from models import *
from mobilenetV2 import mobilenetv2
from train_test import evaluate, evaluate_pertimestep
from utils import *
from fusion import fuse_module_train
import os
import misc
from misc import NativeScalerWithGradNormCount as NativeScaler
from resnet import resnet18, resnet50, resnet34, resnet101
from resnet12 import resnet12
from resnet20 import resnet20
from datasets import build_dataset, build_cifar10_dataset, build_cifar100_dataset
from copy import deepcopy
from snnConverter import *
import glo
from vgg import vgg16_bn_imagenet

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float64)

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
    parser.add_argument('--output_per_timestep', action='store_true', default=False, help='output accuracy per time-step')
    parser.add_argument('--elastic', action='store_true', default=False, help='use elastic SNN')
    args=parser.parse_args()
    
    misc.init_distributed_mode(args)
    device = torch.device("cuda")

    num_classes = {"imagenet":1000,"cifar100":100,"cifar10":10}
    # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    if args.model == "mobilenetv2":
        model = mobilenetv2(pretrained=False)
    elif args.model == "resnet101":
        model = resnet101(pretrained=True)
    elif args.model == "resnet50":
        model = resnet50(pretrained=False)
    elif args.model == "resnet34":
        model = resnet34(pretrained=False) 
    elif args.model == "resnet18":
        model = resnet18(pretrained=False) 
    elif args.model == "vgg16":
        model = vgg16_bn_imagenet(pretrained=False)
    elif args.model == "resnet12":
        model = resnet12(pretrained=False,num_classes=num_classes[args.dataset])
    elif args.model == "resnet20":
        model = resnet20(pretrained=False,num_classes=num_classes[args.dataset])

    glo.set_value("resnet50_lsq_extend_thd_pos", args.model == "resnet50")
    glo.set_value("quan_target_model", args.model)
    model_teacher = deepcopy(model)
    quantized_train_model_fusebn(model,weightBit=args.WeightBit,actBit=args.ActBit)
    fuse_module_train(model)

    # if args.model == "vgg16":
    #     model.features[3] = QuanConv2dFuseBN(m=model.features[3].m, is_first=True, quan_w_fn=LsqQuan(bit=args.WeightBit,all_positive=False,symmetric=False,per_channel=False),
    #                                             quan_a_fn=LsqQuan(bit=args.ActBit-1,all_positive=False,symmetric=False,per_channel=False),
    #                                             quan_out_fn=LsqQuan(bit=args.ActBit-1,all_positive=False,symmetric=False,per_channel=False))

    # if args.model == "vgg16":
    #     model.classifier[0] = QuanLinear(m=model.classifier[0].m, quan_w_fn=LsqQuan(bit=args.WeightBit,all_positive=False,symmetric=False,per_channel=False),
    #                                         quan_out_fn=LsqQuan(bit=args.ActBit,all_positive=True,symmetric=False,per_channel=False))
    #     model.classifier[3] = QuanLinear(m=model.classifier[3].m, quan_w_fn=LsqQuan(bit=args.WeightBit,all_positive=False,symmetric=False,per_channel=False),
    #                                         quan_out_fn=LsqQuan(bit=args.ActBit,all_positive=True,symmetric=False,per_channel=False))

    batch_size = args.batch_size
    data_path = args.dataPath    

    # Load data
    if args.dataset == "imagenet":
        dataset_val = build_dataset(is_train=False, args=args)
    elif args.dataset == "cifar10":
        dataset_val = build_cifar10_dataset(is_train=False, args=args)
    elif args.dataset == "cifar100":
        dataset_val = build_cifar100_dataset(is_train=False, args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val,sampler=sampler_val,
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
    model_without_ddp = model.module
    # apply initializer
    # model.apply(init_weights)
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))
    # fuse_module(model)
    
    #quantized_train_model(model,bit=8)
    model.to(device)
    model.train()
    
    print(model)
    # create criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()

    eff_batch_size = args.batch_size * misc.get_world_size()
    # apply initializer
    state_dict = torch.load(args.quanmodel,map_location=device)
    info = model.module.load_state_dict(state_dict,strict=False)
    print(info)
    f = open("QAT_model.txt","w+")
    f.write(str(model))
    f.close()
    
    set_init_false(model)
    # val_stats = evaluate(data_loader_val, model, device)
    quantized_inference_model_fusebn(model)
    f = open("Inference_model.txt","w+")
    f.write(str(model))
    f.close()

    glo._init()

    output_bin_qann_dir = f"/data1/user/output_bin_qann_{args.model}_w{args.WeightBit}_a{args.ActBit}/"
    
    if args.rank == 0:
        if not os.path.exists(output_bin_qann_dir):
            os.mkdir(output_bin_qann_dir)
        glo.set_value("output_bin_qann_dir",output_bin_qann_dir)
        save_for_bin(model, output_bin_qann_dir)

    if not args.elastic:
        stats = evaluate(data_loader_val,model,device)

    # print(model.module.input.mean())
    # print(model.module.midFeature1.mean())
    # print(model.module.midFeature.mean())
    # print("accuracy:",percent)
    # model = IntegerSNNWarapperElastic(ANNModel=model,bit=args.ActBit, max_timestep=args.maxTimeStep, modelName = args.model, confidence_thr=0.3, true_stop=True)
    if args.elastic:
        model = IntegerSNNWarapperElastic(ANNModel=model,bit=args.ActBit, max_timestep=args.maxTimeStep, modelName = args.model, confidence_thr=0.3, true_stop=True)
    else:
        model = IntegerSNNWarapper(ANNModel=model,bit=args.ActBit, max_timestep=args.maxTimeStep, modelName = args.model)

    f = open("SNN_model.txt","w+")
    f.write(str(model))
    f.close()
    calOrder = []
    set_snn_save_name(model,calOrder)
    
    # output_bin_snn_dir = f"/data1/user/output_bin_snn_{args.model}_w{args.WeightBit}_a{args.ActBit}_T{args.maxTimeStep}/"
    # if args.rank == 0:
    #     if not os.path.exists(output_bin_snn_dir):
    #         os.mkdir(output_bin_snn_dir)
    #     glo.set_value("output_bin_snn_dir",output_bin_snn_dir)

    # if args.rank == 0:
    #     save_for_bin_snn(model,output_bin_snn_dir)

    # if args.rank == 0:
    #     f = open(f"{output_bin_snn_dir}/calculationOrder.txt","w+")
    #     for order in calOrder:
    #         f.write(order+"\n")
    #     f.close()
    
    # # torch.distributed.barrier()
    # print(model)
    if args.output_per_timestep:
        stats = evaluate_pertimestep(data_loader_val,model,device)
    else:
        stats = evaluate(data_loader_val,model,device)

    # print(model.inputAccu.mean())
    # print(model.midFeatureAccu1.mean())
    # print(model.midFeatureAccu.mean())
    
    
            
