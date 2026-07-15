
import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
from const import *
import torch.nn.functional as F



LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, hyp, optimizer, epoch, power=1.1):
    lr = round(hyp['lr'] * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args, train_loader, model, criterion, optimizer, epoch,scaler,verbose=False,ema=None,device=None):
    model.train()
    print("epoch: ", epoch)
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    if verbose:
        LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.to(device).float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        if verbose:
            pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{args.max_epochs - 1}', tversky_loss, focal_loss, loss.item()))
    return ema if ema is not None else None




@torch.no_grad()
def val(val_loader = None, model = None, half = False, args=None, device=None):

    model.eval()
    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    total_batches = len(val_loader)
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.to(device).half() / 255.0 if half else input.to(device).float() / 255.0
        
        input_var = input.to(device)

        with torch.no_grad():
            output = model(input_var)

        out_da = output[0]
        target_da = target[0]

        _,da_predict = torch.max(out_da, 1)
        da_predict = da_predict[:,12:-12]
        _,da_gt=torch.max(target_da, 1)

        DA.reset()
        # print(da_predict.shape, da_gt.shape)
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))

        ###-------------Drivable Segmetation--------------

        ###-------------Lane Segmetation-----------------

        out_ll = output[1]        
        target_ll = target[1]

        _,ll_predict=torch.max(out_ll, 1)
        ll_predict = ll_predict[:,12:-12]
        _,ll_gt=torch.max(target_ll, 1)
        
        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))

        ###-------------Lane Segmetation-----------------
    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    
    return da_segment_result,ll_segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

def train_kd(args, train_loader, model, teacher_model, criterion,
             optimizer, epoch, scaler=None, verbose=False, ema=None):



    device = next(model.parameters()).device
    device_type = device.type
    use_amp = scaler is not None and device_type == "cuda"

    total_batches = len(train_loader)

    meters = {
            # supervised raw
            "tversky_da": AverageMeter(),
            "tversky_ll": AverageMeter(),
            "focal_da": AverageMeter(),
            "focal_ll": AverageMeter(),

            # supervised weighted
            "sup_w": AverageMeter(),

            # logit KD raw
            "kl_da": AverageMeter(),
            "kl_ll": AverageMeter(),

            # logit KD weighted
            "kl_da_w": AverageMeter(),
            "kl_ll_w": AverageMeter(),
            "logit_w": AverageMeter(),

            # feature KD raw
            "enc_0_feat": AverageMeter(),
            "enc_1_feat": AverageMeter(),
            "enc_2_feat": AverageMeter(),
            "encoder_feat_raw": AverageMeter(),

            # feature KD weighted
            "feat_w": AverageMeter(),

            # total
            "total": AverageMeter(),
        }

    pbar = enumerate(train_loader)

    if verbose:
        LOGGER.info(
            ('\n' + '%13s' * 8) %
            ('Epoch', 'Sup', 'KL_DA', 'KL_LL', 'LogitW', 'FeatW', 'Total', 'LR')
        )

        pbar = tqdm( pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}' )

    for i, (_, input_data, target) in pbar:
        input_data = input_data.to(device, non_blocking=True).float() / 255.0

        if isinstance(target, (list, tuple)):
            target = [t.to(device, non_blocking=True) for t in target]
        else:
            raise TypeError(
                "target should be a list/tuple: target[0]=DA mask, target[1]=LL mask"
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_output = teacher_model(input_data)

        student_output = model(input_data)

        ( teacher_out_da_logit, teacher_out_ll_logit, teacher_out_enc_0,teacher_out_enc_1,teacher_out_enc_2 ) = teacher_output

        ( student_out_da_logit, student_out_ll_logit, student_out_enc_0,student_out_enc_1,student_out_enc_2 ) = student_output



        _, da_gt = torch.max(target[0], 1)
        _, ll_gt = torch.max(target[1], 1)

        with torch.cuda.amp.autocast():
            ( tversky_da_loss, tversky_ll_loss, focal_da_loss, focal_ll_loss ) = criterion( [student_out_da_logit, student_out_ll_logit], target )

            sup_loss = (
                args.lambda_tversky_da * tversky_da_loss
                + args.lambda_tversky_ll * tversky_ll_loss
                + args.lambda_focal_da * focal_da_loss
                + args.lambda_focal_ll * focal_ll_loss
            )

            kl_da_logit_loss = kd_logit_loss_weighted(
                student_out_da_logit,
                teacher_out_da_logit.detach(),
                gt_mask=target[0],
                tau=2.0,
                lane_weight=1
            )
            
            kl_ll_logit_loss = bpkd_loss_binary(student_out_ll_logit,
                                                teacher_out_ll_logit.detach(),
                                                ll_gt, tau=1.0, lambda_body = 0.2, lambda_edge = 0.8                  
            )
        
            kl_da_w = args.lambda_kl_da * kl_da_logit_loss
            kl_ll_w = args.lambda_kl_ll * kl_ll_logit_loss

            logit_loss = kl_da_w + kl_ll_w

            enc_0_feat_loss = kd_feat_loss_single(
                    student_out_enc_0,
                    teacher_out_enc_0.detach(),
                    P=4,
                    pooling="avg"
                )
            enc_1_feat_loss = kd_feat_loss_single(
                student_out_enc_1,
                teacher_out_enc_1.detach(),
                P=2,
                pooling="avg"
            )
            enc_2_feat_loss = kd_feat_loss_single(
                student_out_enc_2,
                teacher_out_enc_2.detach(),
                P=1,
                pooling="max"
            )

            encoder_feat_loss = (args.lambda_feat_0 * enc_0_feat_loss + args.lambda_feat_1 * enc_1_feat_loss + args.lambda_feat_2 * enc_2_feat_loss)
            feat_loss = encoder_feat_loss

            loss = sup_loss + logit_loss + feat_loss

        if use_amp:
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = input_data.size(0)

        # supervised raw
        meters["tversky_da"].update(tversky_da_loss.detach().item(), bs)
        meters["tversky_ll"].update(tversky_ll_loss.detach().item(), bs)
        meters["focal_da"].update(focal_da_loss.detach().item(), bs)
        meters["focal_ll"].update(focal_ll_loss.detach().item(), bs)

        # supervised weighted
        meters["sup_w"].update(sup_loss.detach().item(), bs)

        # logit KD raw
        meters["kl_da"].update(kl_da_logit_loss.detach().item(), bs)
        meters["kl_ll"].update(kl_ll_logit_loss.detach().item(), bs)

        # logit KD weighted
        meters["kl_da_w"].update(kl_da_w.detach().item(), bs)
        meters["kl_ll_w"].update(kl_ll_w.detach().item(), bs)
        meters["logit_w"].update(logit_loss.detach().item(), bs)

        # feature KD raw
        meters["enc_0_feat"].update(enc_0_feat_loss.detach().item(), bs)
        meters["enc_1_feat"].update(enc_1_feat_loss.detach().item(), bs)
        meters["enc_2_feat"].update(enc_2_feat_loss.detach().item(), bs)
        meters["encoder_feat_raw"].update(encoder_feat_loss.detach().item(), bs)

        # feature KD weighted
        meters["feat_w"].update(feat_loss.detach().item(), bs)

        # total
        meters["total"].update(loss.detach().item(), bs)

        if verbose:
            lr = optimizer.param_groups[0]["lr"]

            pbar.set_description(
                f"Epoch {epoch} "
                f"sup_w={meters['sup_w'].avg:.4f} "
                f"kl_da_w={meters['kl_da_w'].avg:.4f} "
                f"kl_ll_w={meters['kl_ll_w'].avg:.4f} "
                f"logit_w={meters['logit_w'].avg:.4f} "
                f"feat_w={meters['feat_w'].avg:.4f} "
                f"total={meters['total'].avg:.4f} "
                f"lr={lr:.2e}"
            )

        if i % 400 == 0:
            print(
                f"[Epoch {epoch} Iter {i}/{total_batches}] "
                f"tv_da={meters['tversky_da'].avg:.6f} "
                f"tv_ll={meters['tversky_ll'].avg:.6f} "
                f"fc_da={meters['focal_da'].avg:.6f} "
                f"fc_ll={meters['focal_ll'].avg:.6f} "
                f"sup_w={meters['sup_w'].avg:.6f} | "

                f"kl_da={meters['kl_da'].avg:.6f} "
                f"kl_ll={meters['kl_ll'].avg:.6f} "
                f"kl_da_w={meters['kl_da_w'].avg:.6f} "
                f"kl_ll_w={meters['kl_ll_w'].avg:.6f} "
                f"logit_w={meters['logit_w'].avg:.6f} | "

                f"enc0={meters['enc_0_feat'].avg:.6f} "
                f"enc1={meters['enc_1_feat'].avg:.6f} "
                f"enc2={meters['enc_2_feat'].avg:.6f} "
                f"enc_raw={meters['encoder_feat_raw'].avg:.6f} "
                f"feat_w={meters['feat_w'].avg:.6f} | "

                f"total={meters['total'].avg:.6f}"
            )

    print(
        f"\n[Epoch {epoch} KD Summary] "
        f"tv_da={meters['tversky_da'].avg:.6f} "
        f"tv_ll={meters['tversky_ll'].avg:.6f} "
        f"fc_da={meters['focal_da'].avg:.6f} "
        f"fc_ll={meters['focal_ll'].avg:.6f} "
        f"sup_w={meters['sup_w'].avg:.6f} | "

        f"kl_da={meters['kl_da'].avg:.6f} "
        f"kl_ll={meters['kl_ll'].avg:.6f} "
        f"kl_da_w={meters['kl_da_w'].avg:.6f} "
        f"kl_ll_w={meters['kl_ll_w'].avg:.6f} "
        f"logit_w={meters['logit_w'].avg:.6f} | "

        f"enc0={meters['enc_0_feat'].avg:.6f} "
        f"enc1={meters['enc_1_feat'].avg:.6f} "
        f"enc2={meters['enc_2_feat'].avg:.6f} "
        f"enc_raw={meters['encoder_feat_raw'].avg:.6f} "
        f"feat_w={meters['feat_w'].avg:.6f} | "

        f"total={meters['total'].avg:.6f}\n"
    )

    return ema if ema is not None else None

def kd_logit_loss_class_balanced(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt_mask: torch.Tensor = None,
    tau: float = 2.0,
    lane_alpha: float = 0.5,
    bg_alpha: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Class-balanced KL distillation loss.

    student_logits: [B, 2, H, W]
    teacher_logits: [B, 2, H, W]
    gt_mask: [B, 2, H0, W0], channel 0 = bg, channel 1 = lane

    loss = bg_alpha * mean_KL_bg + lane_alpha * mean_KL_lane
    """

    assert student_logits.shape == teacher_logits.shape
    assert student_logits.dim() == 4
    assert student_logits.size(1) == 2

    B, C, H, W = student_logits.shape

    log_p_s = F.log_softmax(student_logits.float() / tau, dim=1)

    with torch.no_grad():
        p_t = F.softmax(teacher_logits.float() / tau, dim=1)

    # [B, 2, H, W]
    kl_map = F.kl_div(
        log_p_s,
        p_t,
        reduction="none"
    ) * (tau ** 2)

    # [B, H, W]
    kl_map = kl_map.sum(dim=1)

    if gt_mask is None:
        return kl_map.mean()

    assert gt_mask.dim() == 4
    assert gt_mask.size(1) == 2

    # [B, 1, H0, W0]
    lane_mask = gt_mask[:, 1:2].float()

    if lane_mask.shape[-2:] != (H, W):
        lane_mask = F.interpolate(
            lane_mask,
            size=(H, W),
            mode="nearest"
        )

    # [B, H, W]
    lane_mask = lane_mask.squeeze(1)
    lane_mask = (lane_mask > 0.5).float()

    bg_mask = 1.0 - lane_mask

    lane_count = lane_mask.sum()
    bg_count = bg_mask.sum()

    # Mean KL per class
    if lane_count > 0:
        lane_loss = (kl_map * lane_mask).sum() / (lane_count + eps)
    else:
        lane_loss = torch.zeros_like(kl_map.mean())

    if bg_count > 0:
        bg_loss = (kl_map * bg_mask).sum() / (bg_count + eps)
    else:
        bg_loss = torch.zeros_like(kl_map.mean())

    loss = lane_alpha * lane_loss + bg_alpha * bg_loss

    return loss

def kd_logit_loss_weighted(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt_mask: torch.Tensor = None,
    tau: float = 2.0,
    lane_weight: float = 3.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    student_logits: [B, 2, H, W]
    teacher_logits: [B, 2, H, W]
    gt_mask: [B, 2, H, W], channel 0 = bg, channel 1 = lane
    """

    assert student_logits.shape == teacher_logits.shape

    log_p_s = F.log_softmax(student_logits / tau, dim=1)

    with torch.no_grad():
        p_t = F.softmax(teacher_logits / tau, dim=1)

    # [B, 2, H, W]
    kl_map = F.kl_div(
        log_p_s,
        p_t,
        reduction="none"
    ) * (tau ** 2)

    # [B, H, W]
    kl_map = kl_map.sum(dim=1)

    if gt_mask is not None:
        assert gt_mask.dim() == 4
        assert gt_mask.size(1) == 2

        # lấy channel lane
        lane_mask = gt_mask[:, 1:2, :, :].float()  # [B, 1, H, W]

        if lane_mask.shape[-2:] != kl_map.shape[-2:]:
            lane_mask = F.interpolate(
                lane_mask,
                size=kl_map.shape[-2:],
                mode="nearest"
            )

        lane_mask = lane_mask.squeeze(1)  # [B, H, W]

        # background = 1, lane = lane_weight
        weight = 1.0 + (lane_weight - 1.0) * lane_mask

        loss = (kl_map * weight).sum() / (weight.sum() + eps)
    else:
        loss = kl_map.mean()

    return loss

def kd_logit_loss_lane_only(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    KL distillation chỉ trên vùng lane GT.

    student_logits: [B, 2, H, W]
    teacher_logits: [B, 2, H, W]
    gt_mask: [B, 2, H0, W0], channel 0 = bg, channel 1 = lane
    """

    assert student_logits.shape == teacher_logits.shape
    assert student_logits.dim() == 4
    assert student_logits.size(1) == 2
    assert gt_mask.dim() == 4
    assert gt_mask.size(1) == 2

    B, C, H, W = student_logits.shape

    log_p_s = F.log_softmax(student_logits.float() / tau, dim=1)

    with torch.no_grad():
        p_t = F.softmax(teacher_logits.float() / tau, dim=1)

    # [B, 2, H, W]
    kl_map = F.kl_div(
        log_p_s,
        p_t,
        reduction="none"
    ) * (tau ** 2)

    # [B, H, W]
    kl_map = kl_map.sum(dim=1)

    # [B, 1, H0, W0]
    lane_mask = gt_mask[:, 1:2].float()

    if lane_mask.shape[-2:] != (H, W):
        # Nếu logits thấp hơn mask, max_pool giữ lane tốt hơn nearest
        h0, w0 = lane_mask.shape[-2:]
        if H < h0 or W < w0:
            lane_mask = F.adaptive_max_pool2d(lane_mask, output_size=(H, W))
        else:
            lane_mask = F.interpolate(lane_mask, size=(H, W), mode="nearest")

    # [B, H, W]
    lane_mask = lane_mask.squeeze(1)
    lane_mask = (lane_mask > 0.5).float()

    lane_count = lane_mask.sum()


    if lane_count < 1:
        return torch.zeros_like(kl_map.mean())

    loss = (kl_map * lane_mask).sum() / (lane_count + eps)

    return loss


def kd_feat_loss_single(student_feat,
                        teacher_feat,
                        P=1,
                        pooling="avg",
                        normalize=True,
                        eps=1e-8):
    """
    Feature KD loss for lane segmentation.

    student_feat: [B, C, H, W]
    teacher_feat: [B, C, H, W]
    gt_mask: [B, 2, H0, W0], channel 0 = background, channel 1 = lane

    loss_type:
        "pi" : pixel-wise cosine loss
        "pa"  : pair-wise affinity loss
    """

    assert student_feat.dim() == 4
    assert teacher_feat.dim() == 4
    assert student_feat.shape[0] == teacher_feat.shape[0]
    assert student_feat.shape[2:] == teacher_feat.shape[2:]

    teacher_feat = teacher_feat.detach()

    if P > 1:
        if pooling == "avg":
            student_feat = F.avg_pool2d(student_feat, kernel_size=P, stride=P)
            teacher_feat = F.avg_pool2d(teacher_feat, kernel_size=P, stride=P)
        
        elif pooling == "max":
            student_feat = F.max_pool2d(student_feat, kernel_size=P, stride=P)
            teacher_feat = F.max_pool2d(teacher_feat, kernel_size=P, stride=P)

        else:
            raise ValueError("pooling must be 'avg' or 'max'")

    B, Cs, H, W = student_feat.shape
    _, Ct, _, _ = teacher_feat.shape


    # Pair-wise loss
    N = H * W

    # [B, C, N]
    s = student_feat.reshape(B, Cs, N)
    t = teacher_feat.reshape(B, Ct, N)

    if normalize:
        s = F.normalize(s, p=2, dim=1, eps=eps)
        t = F.normalize(t, p=2, dim=1, eps=eps)

    # [B, N, N]
    s_aff = torch.bmm(s.transpose(1, 2), s)
    t_aff = torch.bmm(t.transpose(1, 2), t)


    loss = F.mse_loss(s_aff, t_aff)

    return loss
    
############################################################

def detect_binary_edge(gt):
    """
    Detect thin edge từ binary ground truth.

    gt: [B, H, W] hoặc [B, 1, H, W], giá trị 0/1

    return:
        edge: [B, 1, H, W], float 0/1
    """
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)  # [B,1,H,W]
    elif gt.dim() == 4:
        assert gt.shape[1] == 1, f"Expected [B,1,H,W], got {gt.shape}"
    else:
        raise ValueError(f"Expected gt dim 3 or 4, got {gt.dim()}")

    gt = gt.long()
    edge = torch.zeros_like(gt, dtype=torch.bool)

    # left-right
    diff_w = gt[:, :, :, 1:] != gt[:, :, :, :-1]  # [B,1,H,W-1]
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | diff_w
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | diff_w

    # top-down
    diff_h = gt[:, :, 1:, :] != gt[:, :, :-1, :]  # [B,1,H-1,W]
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | diff_h
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | diff_h

    return edge.float()

def binary_gt_edge_onehot(gt, num_classes=2):
    """
    gt: [B,H,W] or [B,1,H,W]

    return:
        class_edge: [B,C,H,W]
    """
    if gt.dim() == 4:
        gt_2d = gt.squeeze(1)
    else:
        gt_2d = gt

    edge = detect_binary_edge(gt_2d)  # [B,1,H,W]

    onehot = F.one_hot(gt_2d.long(), num_classes=num_classes)  # [B,H,W,C]
    onehot = onehot.permute(0, 3, 1, 2).float()                # [B,C,H,W]

    class_edge = onehot * edge  # [B,C,H,W]

    return class_edge

def dilate_erode_mask(mask, kernel_size=5):
    """
    mask: [B,C,H,W],

    return:
        processed_mask: [B,C,H,W]
    """
    pad = kernel_size // 2

    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    eroded = -F.max_pool2d(
        -mask,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    out = dilated - eroded
    out = out.clamp(0.0, 1.0)

    return out

def make_bpkd_edge_mask_from_binary_gt(
    gt,
    num_classes=2,
    out_size=None,
    kernel_size=5,
    soft=True
):
    """
    Pipeline sát hình:
    GT
    -> Edge Detection
    -> One-Hot Encoding
    -> Dilation - Erosion
    -> GT Mask
    -> AvgPool2D
    -> Edge Mask

    gt: [B,H,W] hoặc [B,1,H,W], binary 0/1

    return:
        edge_mask: [B,C,H_out,W_out]
    """

    # 1. Edge Detection + One-Hot Encoding
    class_edge = binary_gt_edge_onehot(
        gt,
        num_classes=num_classes
    )  # [B,C,H,W]

    # 2. Dilation - Erosion
    edge_mask = dilate_erode_mask(
        class_edge,
        kernel_size=kernel_size
    )  # [B,C,H,W]

    # 3. AvgPool2D về cùng resolution với logits
    if out_size is not None and edge_mask.shape[-2:] != tuple(out_size):
        if soft:
            edge_mask = F.adaptive_avg_pool2d(edge_mask, out_size)
            edge_mask = edge_mask.clamp(0.0, 1.0)
        else:
            edge_mask = F.interpolate(edge_mask, size=out_size, mode="nearest")
            edge_mask = (edge_mask > 0).float()

    return edge_mask

def bpkd_loss_binary(
    student_logits,
    teacher_logits,
    gt,
    tau=4.0,
    kernel_size=5,
    alpha=2.0,
    lambda_body=20.0,
    lambda_edge=50.0,
    eps=1e-6
):
    """
    student_logits: [B,C,H,W]
    teacher_logits: [B,C,H,W]
    gt:             [B,H_gt,W_gt] hoặc [B,1,H_gt,W_gt]
    """

    assert student_logits.shape == teacher_logits.shape

    B, C, H, W = student_logits.shape
    teacher_logits = teacher_logits.detach()

    # =========================
    # 1. Edge mask từ GT
    # =========================
    edge_mask = make_bpkd_edge_mask_from_binary_gt(
        gt=gt,
        num_classes=C,
        out_size=(H, W),
        kernel_size=kernel_size,
        soft=True
    )  # [B,C,H,W], float [0,1]

    edge_mask = edge_mask.to(student_logits.device).float()
    body_mask = (1.0 - edge_mask).clamp(0.0, 1.0)  # [B,C,H,W]

    # =========================================================
    # 2. Edge Loss: Pre-mask -> KL -> Sum(C) -> Post-mask
    # =========================================================
    s_edge_logits = student_logits * edge_mask
    t_edge_logits = teacher_logits * edge_mask

    log_p_s_edge = F.log_softmax(s_edge_logits / tau, dim=1)

    with torch.no_grad():
        p_t_edge = F.softmax(t_edge_logits / tau, dim=1)

    kl_edge = F.kl_div(
        log_p_s_edge,
        p_t_edge,
        reduction="none"
    ) * (tau ** 2)  # [B,C,H,W]

    # Sum(C)
    kl_edge_spatial = kl_edge.sum(dim=1, keepdim=True)  # [B,1,H,W]

    # Repeat(C)
    kl_edge_repeat = kl_edge_spatial.repeat(1, C, 1, 1)  # [B,C,H,W]

    # Post-mask filtering
    kl_edge_post = kl_edge_repeat * edge_mask  # [B,C,H,W]

    # Sum(W x H)
    edge_loss_per_class = kl_edge_post.sum(dim=(2, 3))  # [B,C]

    # Apply alpha
    edge_loss_per_class = alpha * edge_loss_per_class

    # Sum / Mask Pixels
    edge_pixels = edge_mask.sum(dim=(2, 3)) + eps  # [B,C]
    edge_loss_per_class = edge_loss_per_class / edge_pixels

    valid_edge_class = (edge_mask.sum(dim=(2, 3)) > eps).float()

    loss_edge = (
        edge_loss_per_class * valid_edge_class
    ).sum() / (valid_edge_class.sum() + eps)

    # =========================================================
    # 3. Body Loss: dùng mask ngược của edge
    # =========================================================
    s_body_logits = student_logits * body_mask
    t_body_logits = teacher_logits * body_mask

    log_p_s_body = F.log_softmax(s_body_logits / tau, dim=1)

    with torch.no_grad():
        p_t_body = F.softmax(t_body_logits / tau, dim=1)

    kl_body = F.kl_div(
        log_p_s_body,
        p_t_body,
        reduction="none"
    ) * (tau ** 2)  # [B,C,H,W]

    # Sum(C)
    kl_body_spatial = kl_body.sum(dim=1, keepdim=True)  # [B,1,H,W]

    # Repeat(C)
    kl_body_repeat = kl_body_spatial.repeat(1, C, 1, 1)  # [B,C,H,W]

    # Post-mask body
    kl_body_post = kl_body_repeat * body_mask  # [B,C,H,W]

    body_loss_per_class = kl_body_post.sum(dim=(2, 3))  # [B,C]

    body_pixels = body_mask.sum(dim=(2, 3)) + eps
    body_loss_per_class = body_loss_per_class / body_pixels

    valid_body_class = (body_mask.sum(dim=(2, 3)) > eps).float()

    loss_body = (
        body_loss_per_class * valid_body_class
    ).sum() / (valid_body_class.sum() + eps)

    # =========================================================
    # 4. Total BPKD loss
    # =========================================================
    loss = lambda_body * loss_body + lambda_edge * loss_edge

    return loss