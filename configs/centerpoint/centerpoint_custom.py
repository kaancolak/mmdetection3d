_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/models/centerpoint_pillar02_second_secfpn_nus.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    "car",
    "truck",
    "bus",
    "bicycle",
    "pedestrian",
]

data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

out_size_factor = 1
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(
        point_cloud_range=point_cloud_range,
        in_channels=4,
        feat_channels=[32, 32],
        use_voxel_center_z=False),
    pts_middle_encoder=dict(
        in_channels=32),
    pts_backbone=dict(
        in_channels=32,
        layer_strides=[1, 2, 2],),
    pts_neck=dict(
        upsample_strides=[1, 2, 4], ),
    pts_bbox_head=dict(
        tasks=[dict(
            num_class=len(class_names),
            class_names=class_names)],
        bbox_coder=dict(
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor)),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            nms_type='circle',
            out_size_factor=out_size_factor,)))

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

point_load_dim = 5
point_use_dim = [0, 1, 2, 4]

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        bus=4,
        bicycle=6,
        pedestrian=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=point_load_dim,
        use_dim=point_use_dim,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=point_load_dim,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=point_use_dim,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=point_load_dim,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=point_use_dim,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D')
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=dict(classes=class_names),
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            box_type_3d='LiDAR',
            backend_args=backend_args)))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))

train_cfg = dict(val_interval=5)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_optimizer=True))
