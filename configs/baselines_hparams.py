class BC():
    def __init__(self):
        super(BC, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'lr_scheduler': 'cosine',      # CosineAnnealingLR, T_max = num_epochs
            'image_size': 224,
            'val_split': 0.15,
            'focal_gamma': 2.0,
        }
        self.alg_hparams = {
            # ---- Gradient-reversal adversarial alignment ----
            'DANN': {
                'backbone': 'efficientnet_b0',
                'learning_rate': 1e-3,
                'lambda_gamma': 10.0,          # GRL schedule: 2/(1+exp(-gamma*p)) - 1
                'disc_hidden': 256,
                'disc_dropout': 0.5,
            },

            # ---- Conditional adversarial (CDAN+E) ----
            'CDAN': {
                'backbone': 'efficientnet_b0',
                'learning_rate': 1e-3,
                'lambda_gamma': 10.0,          # GRL warmup steepness
                'disc_hidden': 256,
                'disc_dropout': 0.5,
                'multilinear_dim': 1024,       # random multilinear map dim
                'use_entropy': True,           # entropy weighting -> CDAN+E
            },

            # ---- Deep CORAL (second-order feature alignment) ----
            'Deep_Coral': {
                'backbone': 'efficientnet_b0',
                'learning_rate': 1e-3,
                'coral_wt': 100.0,             # lambda_CORAL
            },

            # ---- DAMS (MMD + CORAL) ----
            'DAMS': {
                'backbone': 'efficientnet_b0',
                'learning_rate': 1e-3,
                'mmd_wt': 1.0,                 # lambda_MMD
                'coral_wt': 100.0,             # lambda_CORAL
            },

            # ---- MemSAC (adversarial + memory-bank supervised contrastive) ----
            'MemSAC': {
                'backbone': 'efficientnet_b0',
                'learning_rate': 1e-3,
                'lambda_gamma': 10.0,          # GRL domain-adversarial schedule
                'disc_hidden': 256,
                'disc_dropout': 0.5,
                'proj_dim': 128,
                'memory_size': 4096,
                'temperature': 0.07,
                'contrast_wt': 1.0,            # lambda_contrast
                'pseudo_threshold': 0.9,
                'contrast_warmup_frac': 0.1,
            },
        }
