import sys
import os
import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ddpm import Unet3D, GaussianDiffusion, Trainer
import importlib


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, cfg, world_size):
    if cfg.dataset.dist:
        setup(rank, world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    # 初始化模型
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.dataset.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            out_dim=cfg.model.out_dim,
            cond_dim=cfg['dataset'].get('cond_dim'),
        ).to(device)
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    if cfg.dataset.dist:
        model = DDP(model, device_ids=[rank])

    # 初始化 diffusion 过程
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.dataset.diffusion_img_size,
        num_frames=cfg.dataset.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).to(device)

    dataset_module = importlib.import_module(f'dataset.{cfg.dataset.name}_dataloader')
    train_dataloader = dataset_module.get_loader(cfg.dataset, rank)

    # 初始化训练器
    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataloader,
        train_batch_size=cfg.model.batch_size // world_size if cfg.dataset.dist else cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
    )

    # 如果指定了里程碑模型，则加载模型
    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    # 开始训练
    trainer.train()

    if cfg.dataset.dist:
        cleanup()


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def main(cfg: DictConfig):
    world_size = len(cfg.model.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.model.gpus))

    if cfg.dataset.dist:
        mp.spawn(train, args=(cfg, world_size), nprocs=world_size, join=True)
    else:
        train(0, cfg, world_size)


if __name__ == '__main__':
    main()
