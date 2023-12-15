#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
import os
import random
from random import randint
import uuid
import time
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.h5 import H5
from utils.dataset import CamInfoDataset, ObjDict
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()

    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1

    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             load_iteration=None, shuffle=True, load_images_mode=2):
    """
    Args added - defaults match original behaviour
        load_iteration  (int [None]) -1 loads latst point_cloud
        shuffle         (bool [True])

    low memory args added:
        load_images_mode    (int)   1: original default behaviour, load image to gpu
                                    2: creates h5 file
                                    0: load on training loop
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=shuffle, load_images_mode=load_images_mode)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    h5name = None
    images_path = None
    dbname = "images" if dataset.resolution in (0,1) else f"images_{dataset.resolution}"

    if load_images_mode == 0: # load on training loop
        images_path = os.path.join(dataset.source_path, dbname)
        dset = CamInfoDataset(scene.getTrainCameras(), images_path)
        dload = DataLoader(dset, batch_size=1, shuffle=True, pin_memory=True, num_workers=8)
        data_iterator = iter(dload)
    elif load_images_mode == 2: # load from h5 file
        h5name = os.path.join(dataset.source_path, "images.h5")

    with H5(h5name) as hf: # acts as paththrough if no h5name
        if hf.file is not None:
            names = [os.path.splitext(name)[0] for name in hf.get_text(dbname='names')]

        for iteration in range(first_iter, opt.iterations + 1):
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if load_images_mode in (1,2): # viewpoint_stack in memory
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            else:  # viewpoint_stack in Dataset
                if data_iterator._num_yielded >= len(data_iterator):
                    data_iterator = iter(dload)
                cam = next(data_iterator)
                viewpoint_cam = ObjDict(original_image=cam[0], world_view_transform=cam[1],
                                        full_proj_transform=cam[2], camera_center=cam[3],
                                        FoVx=cam[4], FoVy=cam[5], image_width=cam[6],
                                        image_height=cam[7])

            if load_images_mode != 1: # viewpoint_stack stored in cpu
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

            # Pick a random high resolution camera
            if random.random() < 0.3 and dataset.sample_more_highres:
                print(" is this being called?")
                viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # Load images to ensure viewpoint_cam hasc coorect dimensions
            # to prevent possible image size rounding errors on downscaling
            if load_images_mode < 2:
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = hf.get_image(names.index(viewpoint_cam.image_name),
                                        dbname=dbname, device="cuda")

            viewpoint_cam.image_height, viewpoint_cam.image_width = gt_image.shape[-2:]

            #TODO ignore border pixels
            if dataset.ray_jitter:
                subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
                # subpixel_offset *= 0.0
            else:
                subpixel_offset = None
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            # sample gt_image with subpixel offset
            if dataset.resample_gt_image:
                gt_image = create_offset_gt(gt_image, subpixel_offset)

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        gaussians.compute_3D_filter(cameras=trainCameras)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                    if iteration < opt.iterations - 100:
                        # don't update in the end of training
                        gaussians.compute_3D_filter(cameras=trainCameras)
            
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def train_log(_dataset, _opt, _pipe, _args):

    print('dataset = lp.extract(args)')
    for k,v in _dataset.__dict__.items():
        print(f"  {k:<24}{v}")

    print('opt = op.extract(args)')
    for k,v in _opt.__dict__.items():
        print(f"  {k:<24}{v}")

    print('pipe = pp.extract(args)')
    for k,v in _pipe.__dict__.items():
        print(f"  {k:<24}{v}")

    print(f'testing_iterations:    {_args.test_iterations}')
    print(f'saving_iterations:     {_args.save_iterations}')
    print(f'checkpoint_iterations: {_args.checkpoint_iterations}')
    print(f'debug_from:            {_args.debug_from}')

    print(f'load_iteration         {_args.load_iteration}')
    print(f'shuffle                {_args.shuffle}')
    print(f'load_images_mode       {_args.load_images_mode}')

    print(f'ip:                    {_args.ip}')
    print(f'port:                  {_args.port}')
    print(f'detect_anomaly:        {_args.detect_anomaly}')
    print(f'quiet:                 {_args.quiet}')
    print(f'only_log:              {_args.only_log}')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # enable ray_jitter as arg, set default to False
    # parser.add_argument("--ray_jitter", action='store_true', default=False)
    # lower_memory options:
    # load_images_mode: default 1 load to memory as original, 0, load on train loop 2, .h5
    parser.add_argument("--load_images_mode", type=int, default=2)
    # scripted options
    parser.add_argument("--load_iteration", type=int, default = None)
    # shuffle is default true
    parser.add_argument("--shuffle", action='store_false', default=True)
    # added to inspect training arguments
    parser.add_argument("--only_log", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    train_log(dataset, opt, pipe, args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if not args.only_log:
        _time = time.time()
        training(dataset, opt, pipe, args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
                 args.load_iteration, args.shuffle, args.load_images_mode)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # All done
        print(f"\nTraining complete. {round(time.time()-_time, 3)}s")
