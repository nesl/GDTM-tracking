_base_ = [
    '../_base_/datasets/ucla_sample_1car.py'
]

trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_train/',
        num_future_frames=0,
        num_past_frames=9,
        valid_nodes=[1,2,3],
        valid_mods=['mocap', 'realsense_camera_img', 'realsense_camera_depth'],
        min_x=-1.5, max_x=2.5,
        min_y=0, max_y=1,
        min_z=-1.5, max_z=2.5,
        include_z=True,
    ),
    num_future_frames=0,
    num_past_frames=1,
)

valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_val/',
        valid_nodes=[1,2,3],
        valid_mods=['mocap', 'realsense_camera_img', 'realsense_camera_depth'],
        min_x=-1.5, max_x=2.5,
        min_y=0, max_y=1,
        min_z=-1.5, max_z=2.5,
        include_z=True,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)

testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_test/',
        min_x=-1.5, max_x=2.5,
        min_y=0, max_y=1,
        min_z=-1.5, max_z=2.5,
        valid_nodes=[1,2,3],
        valid_mods=['mocap', 'realsense_camera_img', 'realsense_camera_depth'],
        include_z=True,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)

backbone_cfg_img = [
    dict(type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1
    )
]

backbone_cfg_depth = [
    dict(type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1
    )
]


model_cfg_img=dict(type='LinearEncoder', in_len=135, out_len=1,
        ffn_cfg=dict(type='SLP', in_channels=256))

model_cfg_depth = dict(type='LinearEncoder', in_len=108, out_len=1,
        ffn_cfg=dict(type='SLP', in_channels=256))


model_cfgs = {('realsense_camera_img'): model_cfg_img,
              ('realsense_camera_depth'): model_cfg_depth
            }


backbone_cfgs = {'realsense_camera_img': backbone_cfg_img, 'realsense_camera_depth' : backbone_cfg_depth
}

model = dict(type='KFDETR',
        output_head_cfg=dict(type='OutputHead',
         include_z=True,
         predict_full_cov=True,
         cov_add=1.0,
         input_dim=256,
         predict_rotation=True,
         predict_velocity=False,
         num_sa_layers=0,
         to_cm=True,
         mlp_dropout_rate=0.3
    ),
    model_cfgs=model_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    pos_loss_weight=1,
    num_queries=1,
    mod_dropout_rate=0.3,
    loss_type='nll'
)


# orig_bs = 2
# orig_lr = 1e-4 
# factor = 4
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=testset
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    # paramwise_cfg=dict(
        # custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            # 'sampling_offsets': dict(lr_mult=0.1),
            # 'reference_points': dict(lr_mult=0.1)
        # }
    # )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 50
lr_config = dict(policy='step', step=[40])
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

find_unused_parameters = True

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
