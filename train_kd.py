import os
import torch
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler, train_kd
import torch.optim.lr_scheduler
from copy import deepcopy
import math
from criterion import TotalLoss
import yaml
def make_resume_checkpoint(epoch, model, ema, optimizer, scaler, lr, use_ema):
    """
    Full checkpoint dùng để resume training.
    """
    return {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.ema.state_dict() if use_ema else None,
        "updates": ema.updates if use_ema else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "lr": lr
    }
class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
def load_teacher_weights(teacher_model, teacher_path, device):
    print(f"=> Loading teacher checkpoint '{teacher_path}'")

    checkpoint = torch.load(teacher_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    missing_keys, unexpected_keys = teacher_model.load_state_dict(checkpoint, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    teacher_model.eval()

    for p in teacher_model.parameters():
        p.requires_grad = False

    return teacher_model
def load_student_weights(student_model, student_path, device, strict=False):
    print(f"=> Loading student checkpoint '{student_path}'")

    checkpoint = torch.load(student_path, map_location=device)

    # Case 1: full checkpoint dạng .pth.tar
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Case 2: best_*.pth chỉ là state_dict
    missing_keys, unexpected_keys = student_model.load_state_dict(
        checkpoint,
        strict=strict
    )

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Student vẫn train tiếp
    student_model.train()

    for p in student_model.parameters():
        p.requires_grad = True

    print("netParams: ",netParams(student_model))

    return student_model
def train_net(args):
    use_ema = args.ema
    # load the model
    cuda_available = torch.cuda.is_available()
    
    if args.model == "twinmixing":
        with open("hyperparameters/twinmixing_hyper.yaml", errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        from TwinMixing import twinmixing as net
        model = net.TwinMixing(args, student=True)
        model =  load_student_weights(model, "pretrained/twinmixing_nano.pth","cuda")
        teacher_model = net.TwinMixing(args, student=False)
        teacher_model =  load_teacher_weights(teacher_model, "pretrained/twinmixing_large.pth","cuda")
    elif args.model == "twinplus":
        with open("hyperparameters/twinplus_hyper.yaml", errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        from TwinPlus import model as net
        model = net.TwinLiteNetPlus(args, student=True)
        model =  load_student_weights(model, "pretrained/twinplus_nano.pth","cuda")
        teacher_model = net.TwinLiteNetPlus(args, student=False)
        teacher_model =  load_teacher_weights(teacher_model, "pretrained/twinplus_large.pth","cuda")


    args.savedir = args.savedir

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(args.data_dir, hyp, valid=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(args.data_dir, hyp, valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        
    device = None
    if cuda_available:
        device = torch.device("cuda")
        args.onGPU = True

        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        
        model = model.to(device)
        cudnn.benchmark = True
    else:
        args.onGPU = False
        device = torch.device("cpu")
    print(device)
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))


    criteria = TotalLoss(hyp)

    start_epoch = 0
    lr = hyp['lr']

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
    if use_ema:
        ema = ModelEMA(model)
    if args.resume:
        if os.path.isfile(args.resume):
            if args.resume.split(".")[-1] == "tar":
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                ema.ema.load_state_dict(checkpoint['ema_state_dict'])
                ema.updates=checkpoint['updates']
                print(ema.updates)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scaler = torch.cuda.amp.GradScaler()

    da_segment_results,ll_segment_results = val(valLoader, ema.ema if use_ema else model,args=args, device=device) #da_mIoU_seg, ll_IoU_seg
    msg =  'Driving area Segment: mIOU({da_seg_miou:.3f})\n' \
                    'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})'.format(
                        da_seg_miou=da_segment_results[2],
                        ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1])
    print("Pretrained model evaluation:")
    print(msg)
    
    import time
    
    for epoch in range(start_epoch, args.max_epochs):
        
        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args,hyp,optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        teacher_model.eval()
        start = time.time()
        ema = train_kd(args, trainLoader, model, teacher_model, criteria, optimizer, epoch,scaler,args.verbose,ema if use_ema else None)
        print("Epoch training time: {:.3f} seconds".format(time.time() - start))
        

        model.eval()
        start = time.time()
        da_segment_results,ll_segment_results = val(valLoader, ema.ema if use_ema else model,args=args, device=device) #da_mIoU_seg, ll_IoU_seg
        print("Epoch validation time: {:.3f} seconds".format(time.time() - start))
        msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                    'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                        da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                        ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
        print(msg)
        torch.save(ema.ema.state_dict() if use_ema else model.state_dict(), model_file_name)
        last_checkpoint = make_resume_checkpoint( epoch=epoch, model=model, ema=ema, optimizer=optimizer, scaler=scaler, lr=lr, use_ema=use_ema)
        save_checkpoint(last_checkpoint, os.path.join(args.savedir, "last.pth.tar"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=30, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=16, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--savedir', default='out_kd/', help='directory to save the results')
    parser.add_argument('--model', help='Model architecture')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument("--teacher-type", default="large", help="Model configuration")
    parser.add_argument("--student-type", default="nano",  help="Model configuration")
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--ema', action='store_true', help='')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Dataset directory')


    parser.add_argument("--lambda_tversky_da", type=float, default=1.)
    parser.add_argument("--lambda_focal_da", type=float, default=1.)

    parser.add_argument("--lambda_tversky_ll", type=float, default=1.) # 1.2
    parser.add_argument("--lambda_focal_ll", type=float, default=1.)

    parser.add_argument("--lambda_kl_da", type=float, default=0.65) 
    parser.add_argument("--lambda_kl_ll", type=float, default=0.35) 

    parser.add_argument("--lambda_feat_0", type=float, default=0.05)
    parser.add_argument("--lambda_feat_1", type=float, default=0.05) 
    parser.add_argument("--lambda_feat_2", type=float, default=0.5) 
    args = parser.parse_args()

    train_net(args)
