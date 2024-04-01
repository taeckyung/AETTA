
args = None

CIFAR10Opt = {
    'name': 'cifar10',
    'batch_size': 64, # 128

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
                    "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
    
    'indices_in_1k' : None
}

CIFAR100Opt = {
    'name': 'cifar100',
    'batch_size': 128,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-100-C',
    'classes': ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                'bottles', 'bowls', 'cans', 'cups', 'plates',
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow',
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'num_class': 100,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
    
    'indices_in_1k' : None
}

IMAGENET_C = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/ImageNet-C',
    'num_class': 1000,
    'severity': 5,
    'domains': ["gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],

    'src_domains': ["original"],
    'tgt_domains': ["gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
    
    'indices_in_1k' : None
}

MNISTOpt = {
    'name': 'mnist',
    'batch_size': 128,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/MNIST-C',
    'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'num_class': 10,
    'domains': ["original",

                "test",

                "shot_noise",
                "impulse_noise",
                "glass_blur",
                "motion_blur",
                "shear",
                "scale",
                "rotate",
                "brightness",
                "translate",
                "stripe",
                "fog",
                "spatter",
                "dotted_line",
                "zigzag",
                "canny_edges",

                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "shot_noise",
        "impulse_noise",
        "glass_blur",
        "motion_blur",
        "shear",
        "scale",
        "rotate",
        "brightness",
        "translate",
        "stripe",
        "fog",
        "spatter",
        "dotted_line",
        "zigzag",
        "canny_edges",
    
    ],
    
    'indices_in_1k' : None

}

CONT_SEQUENCE = {
    0 :  ["gaussian_noise-5", "shot_noise-5", "impulse_noise-5", "defocus_blur-5", "glass_blur-5", "motion_blur-5", "zoom_blur-5", "snow-5", "frost-5", "fog-5", "brightness-5", "contrast-5", "elastic_transform-5", "pixelate-5", "jpeg_compression-5"],
    1 :  ['brightness-5', 'pixelate-5', 'gaussian_noise-5', 'motion_blur-5', 'zoom_blur-5', 'glass_blur-5', 'impulse_noise-5', 'jpeg_compression-5', 'defocus_blur-5', 'elastic_transform-5', 'shot_noise-5', 'frost-5', 'snow-5', 'fog-5', 'contrast-5'],
    2  : ['jpeg_compression-5', 'shot_noise-5', 'zoom_blur-5', 'frost-5', 'contrast-5', 'fog-5', 'defocus_blur-5', 'elastic_transform-5', 'gaussian_noise-5', 'brightness-5', 'glass_blur-5', 'impulse_noise-5', 'pixelate-5', 'snow-5', 'motion_blur-5'],
    3  : ['contrast-5', 'defocus_blur-5', 'gaussian_noise-5', 'shot_noise-5', 'snow-5', 'frost-5', 'glass_blur-5', 'zoom_blur-5', 'elastic_transform-5', 'jpeg_compression-5', 'pixelate-5', 'brightness-5', 'impulse_noise-5', 'motion_blur-5', 'fog-5'],
    4  : ['shot_noise-5', 'fog-5', 'glass_blur-5', 'pixelate-5', 'snow-5', 'elastic_transform-5', 'brightness-5', 'impulse_noise-5', 'defocus_blur-5', 'frost-5', 'contrast-5', 'gaussian_noise-5', 'motion_blur-5', 'jpeg_compression-5', 'zoom_blur-5'],
    5  : ['pixelate-5', 'glass_blur-5', 'zoom_blur-5', 'snow-5', 'fog-5', 'impulse_noise-5', 'brightness-5', 'motion_blur-5', 'frost-5', 'jpeg_compression-5', 'gaussian_noise-5', 'shot_noise-5', 'contrast-5', 'defocus_blur-5', 'elastic_transform-5'],
    6  : ['motion_blur-5', 'snow-5', 'fog-5', 'shot_noise-5', 'defocus_blur-5', 'contrast-5', 'zoom_blur-5', 'brightness-5', 'frost-5', 'elastic_transform-5', 'glass_blur-5', 'gaussian_noise-5', 'pixelate-5', 'jpeg_compression-5', 'impulse_noise-5'],
    7  : ['frost-5', 'impulse_noise-5', 'jpeg_compression-5', 'contrast-5', 'zoom_blur-5', 'glass_blur-5', 'pixelate-5', 'snow-5', 'defocus_blur-5', 'motion_blur-5', 'brightness-5', 'elastic_transform-5', 'shot_noise-5', 'fog-5', 'gaussian_noise-5'],
    8  : ['defocus_blur-5', 'motion_blur-5', 'zoom_blur-5', 'shot_noise-5', 'gaussian_noise-5', 'glass_blur-5', 'jpeg_compression-5', 'fog-5', 'contrast-5', 'pixelate-5', 'frost-5', 'snow-5', 'brightness-5', 'elastic_transform-5', 'impulse_noise-5'],
    9  : ['glass_blur-5', 'zoom_blur-5', 'impulse_noise-5', 'fog-5', 'snow-5', 'jpeg_compression-5', 'gaussian_noise-5', 'frost-5', 'shot_noise-5', 'brightness-5', 'contrast-5', 'motion_blur-5', 'pixelate-5', 'defocus_blur-5', 'elastic_transform-5'],
    10  : ['contrast-5', 'gaussian_noise-5', 'defocus_blur-5', 'zoom_blur-5', 'frost-5', 'glass_blur-5', 'jpeg_compression-5', 'fog-5', 'pixelate-5', 'elastic_transform-5', 'shot_noise-5', 'impulse_noise-5', 'snow-5', 'motion_blur-5', 'brightness-5'],
    
    
    
}


IMAGENET_A = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet-a',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/imagenet-a',
    'num_class': 200, # select 200 from 1000
    'severity': 5,

    'domains': ['original', 'corrupt'], 

    'src_domains': ["original"],
    'tgt_domains': ["corrupt"],

    'indices_in_1k' : [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988]

}


IMAGENET_R = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet-r',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/imagenet-r',
    'num_class': 200, # select 200 from 1000
    'severity': 5,

    'domains': ['original', 'corrupt'], 

    'src_domains': ["original"],
    'tgt_domains': ["corrupt"],
    
    'indices_in_1k' : [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]

}


