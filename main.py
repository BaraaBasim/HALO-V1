import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torch
# import matlab.engine
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from tqdm import trange
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import yolov5.val as validate  # for end-of-epoch mAP
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import Model
from yolov5.utils.autoanchor import check_anchors
from yolov5.utils.autobatch import check_train_batch_size
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.downloads import attempt_download, is_url
from yolov5.utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from yolov5.utils.loggers import Loggers
from yolov5.utils.loggers.comet.comet_utils import check_comet_resume
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import fitness
from yolov5.utils.plots import plot_evolve
from yolov5.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
from yolov5.models.common import DetectMultiBackend
                               
# -------------------------------------------
from utils import PatchTransformer, PatchApplier
import val_patch
import torchvision.transforms as TF
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
# from script import generatePSF
from pso_psf import OptimizeFunction, PSO
# -------------------------------------------



LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../v0/config/data_config_all.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default="../v0/checkpoints/yolov5s.pt", help='model path(s)')
    parser.add_argument('--cfg', type=str, default='../v0/yolov5/models/yolov5s.yaml', help='model.yaml')
    parser.add_argument('--hyp', type=str, default='../v0/yolov5/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    
    opt = parser.parse_args() 
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # # Adversarial patch
    # patch_w, patch_h = 300, 300
    # adv_patch = torch.rand(3, patch_w, patch_h).to(device)  # 第一层为rgb
    # adv_patch.requires_grad_(True)
    # optimizer = torch.optim.Adam([adv_patch], lr = 1e-2, amsgrad=True)
    # scheduler_factory = lambda x: lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
    # scheduler = scheduler_factory(optimizer)

    # Create model
    hyp = opt.hyp
    hyp = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    weights = opt.weights
    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)  # create
    exclude = []   # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # Batch size
    batch_size = opt.batch_size

    # Dataloader
    data_dict = check_dataset(opt.data)
    print(data_dict)
    train_path, val_path = data_dict['train'], data_dict['val']
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              opt.batch_size // WORLD_SIZE,
                                              gs,
                                              opt.single_cls,
                                              hyp=opt.hyp,
                                              augment=False,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   opt.batch_size // WORLD_SIZE * 2,
                                   gs,
                                   opt.single_cls,
                                   hyp=opt.hyp,
                                   cache=None if opt.noval else opt.cache,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]
    
    model.eval()
    
    nb = len(train_loader)
    
    # PSO
    pso = PSO(100, device)
    func = OptimizeFunction(model, device)
    
    save_dir = Path('results')
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    best_fitness = 100.0

    
    for epoch in trange(opt.epochs, desc="Epochs counter"):  # epoch ------------------------------------------------------------------
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, desc="Batch counter",total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            targets = targets.to(device)
            imgs = imgs.to(device, non_blocking=True).float() / 255      # (n, c, h, w)
            
            func.set_para(targets, imgs)
            pso.optimize(func)
            swarm_parameters = pso.run()
            # break
            with open('results/log.txt', "a+") as f:
                f.write(f"Epoch: {epoch} Batch: {i} gbest_value: {swarm_parameters.gbest_value}\n")
        
            
            
        # val
        results, maps, _ = val_patch.run(data_dict,
                                         swarm_parameters=swarm_parameters,
                                         batch_size=batch_size // WORLD_SIZE * 2,
                                         imgsz=imgsz,
                                         model=model,
                                         single_cls=opt.single_cls,
                                         dataloader=val_loader,
                                         plots=False
                                         )
        
        print('gbest_position: ', swarm_parameters.gbest_position[0])
        print(swarm_parameters.gbest_position)
        print('gbest_value: ', swarm_parameters.gbest_value)
        with open('results/gbest_position.txt', "a+") as f:
            f.write(f"Epoch: {epoch}, gbest_position: {swarm_parameters.gbest_position}\n")
