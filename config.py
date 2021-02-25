class Cfg:
    """Main config."""

    seed = 0xFACED  # RANDOM SEED
    NUMCLASSES = 5  # CONST

    # MAIN SETTINGS
    experiment_name = "experiment"   # Name of the current approach
    debug = False                    # Debug flag. If set to True,
                                     # the number of samples will be decreased for faster experiments.
    path_to_imgs = "PATH"            # Path to folder with train images
    path_to_csv = "PATH"             # Path to csv-file with targets
    path = "PATH"                    # Working directory,
                                     # the place where the logs and the weigths will be saved
    log = "log.txt"                  # If exists, all the logs will be saved in this txt-file as well
    chk = ""                         # Path to model checkpoint (weights).
                                     # If exists, the model will be uploaded from this checkpoint.
    device = "cuda"                  # Device

    # MODEL'S SETTINGS
    model_name = "/timm/resnet18"  # PyTorch (/torch/) or timm (/timm/) model's name
    pretrained = True              # Pretrained flag

    # TRAINING SETTINGS
    num_epochs = 30        # Number of epochs (including warm-up ones)
    warmup_epochs = 3      # Number of warm-up epochs
    train_batchsize = 128  # Train Batch Size
    train_shuffle = True   # Shuffle during training
    val_batchsize = 128    # Validation Batch Size
    val_shuffle = False    # Shuffle during validation
    test_batchsize = 1     # Test Batch Size
    test_shuffle = False   # Shuffle during testing
    num_workers = 8        # Number of workers (dataloader parameter)
    verbose = True         # If set to True, draws plots of loss changes and validation metric changes.
    early_stopping = 8     # Interrupts training after a certain number of epochs if the metrics stops increasing,
                           # set the value to "-1" if you want to turn off this option.
    savestep = 5           # Number of epochs in loop before saving model.
                           # Example: 10 means that weights will be saved each 10 epochs.

    # VALIDATION STRATEGY SETTINGS
    kfold = True   # Uses Startified K-fold validation strategy if turned on. 
                   # Otherwise, simple train-test split will be used.
    n_splits = 5   # Number of splits for Stratified K-fold
    fold = 1       # Number of fold to train
    train_size, val_size, test_size = 0.8, 0.1, 0.1  # Sizes for train-test split.
                                                     # You can set 0.0 value for testsize,
                                                     # in this case test won't be used.

    # APEX PARAMETERS
    mixed_precision = True         # Mixed precision training flag
    gradient_accumulation = True   # Gradient accumulation flag
    iters_to_accumulate = 8        # Parameter for gradient accumulation

    # TRANSFORMS AND AUGMENTATIONS SETTINGS

    # Progressive resize parameters
    start_size = 256  # Start size for progressive resize
    final_size = 512  # Maximum size for progressive resize
    size_step = 32    # Number to increase image size on each epoch
    # Set the same values for start_size and final_size if you wan't to turn of progressive resize.

    pretransforms = [  # Pre-transforms
        dict(
            name="Resize",
            params=dict(
                height=512,
                width=512,
                p=1.0,
            )
        ),
    ]

    augmentations = [    # Augmentations
        dict(
            name="HorizontalFlip",
            params=dict(
                p=0.5,
            )
        ),
        dict(
            name="VerticalFlip",
            params=dict(
                p=0.5,
            )
        ),
        dict(
            name="Rotate",
            params=dict(
                limit=[-180, 180],
                p=0.5,
            )
        ),
        dict(
            name="RandomBrightnessContrast",
            params=dict(
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5, 
            )
        ),
        dict(
            name="ShiftScaleRotate",
            params=dict(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5, 
            )
        ),
        dict(
            name="HueSaturationValue",
            params=dict( 
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5,
            )
        ),
        dict(
            name="Blur",
            params=dict( 
                blur_limit=2.5,
                p=0.5,
            )
        ),
    ]

    posttransforms = [  # Post-transforms
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        ),
        dict(
            name="/custom/totensor",
            params=dict(
            )
        ),
    ]

    # OPTIMIZER SETTINGS
    optimizer = "/custom/Ranger"  # PyTorch optimizer or custom one
    optimizer_params = dict(
        lr=0.00003,  # Learning rate
    )

    # SCHEDULER SETTINGS
    scheduler = "/custom/CosineBatchDecayScheduler" # PyTorch scheduler or custom one
    scheduler_params = dict(
        epochs=120,
        steps=None,   # Sets in the datagenerator functions according number of samples
        batchsize=train_batchsize,
        decay=32,
        startepoch=4,
        minlr=1e-8,
    )


    # LOSS FUNCTIONS SETTINGS
    lossfn = "/custom/LabelSmoothingLoss"  # PyTorch loss fucntion or custom on
    lossfn_params = dict(
        classes=5,
        smoothing=0.1,
    )

    # DON'T CHANGE
    # Can be changed only with uploading the model from the checkpoint.
    stopflag = 0
    scheduler_state = None
    optim_dict = None
    epoch = 0