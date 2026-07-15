import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
from argparse import ArgumentParser

from utils import val, netParams
import DataSet as myDataLoader

def load_checkpoint(model, weight_path, device, strict=False):
    print(f"=> Loading checkpoint: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return model
def validation(args):
    """
    Perform model validation on the BDD100K dataset.
    :param args: Parsed command-line arguments.
    """
    
    # Initialize model
    if args.model == "twinmixing":
        with open("hyperparameters/twinmixing_hyper.yaml", errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        from TwinMixing import twinmixing as net
        model = net.TwinMixing(args, student=True)
        model = load_checkpoint( model=model, weight_path=args.weight, device="cuda", strict=True)
    elif args.model == "twinplus":
        with open("hyperparameters/twinplus_hyper.yaml", errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        from TwinPlus import model as net
        model = net.TwinLiteNetPlus(args, student=True)
        model = load_checkpoint( model=model, weight_path=args.weight, device="cuda", strict=True)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True

    
    # Create validation data loader
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(args.data_dir, hyp, valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Print model parameter count
    print(f'Total network parameters: {netParams(model)}')
    
    # Load pretrained weights
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    
    # Perform validation
    da_segment_results, ll_segment_results = val(valLoader, model, args.half, args=args,device="cuda")
    
    # Print results
    print(f"Driving Area Segment: mIOU({da_segment_results[2]:.3f})")
    print(f"Lane Line Segment: Acc({ll_segment_results[0]:.3f}) IOU({ll_segment_results[1]:.3f})")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default="pretrained/large.pth", help='Path to model weights')
    parser.add_argument('--model', help='Model architecture')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel threads')
    parser.add_argument('--student_type', type=str, default="nano", help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], help='Model configuration')
    parser.add_argument('--half', action='store_true', help='Use half precision for inference')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Dataset directory')
    
    # Parse arguments and run validation
    validation(parser.parse_args())
