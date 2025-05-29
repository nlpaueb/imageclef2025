import os
import tensorflow as tf
from tagcxn import TagCXN



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

configuration = {
    'data': {
        'train_images_folder': '/media/SSD_2TB/imageclef2025/images/train/',
        'val_images_folder':   '/media/SSD_2TB/imageclef2025/images/valid/',
        'test_images_folder':  '/media/SSD_2TB/imageclef2025/images/dev/',
        
        # 'train_data_path':     '/home/annachatz/5_train_concepts.csv',
        # 'train_data_path':     '/home/annachatz/imageclef2025/merged_concepts_train_val.csv',
        'train_data_path': '/media/SSD_2TB/imageclef2025/splits/processed/train_concepts.csv',
        # 'train_data_path': '/home/annachatz/5_train_concepts.csv',
        # 'val_data_path': '/home/annachatz/5_val_concepts.csv',
        # 'val_data_path':       '/media/SSD_2TB/imageclef2025/splits/processed/dev_concepts.csv',
        'val_data_path': '/media/SSD_2TB/imageclef2025/splits/processed/valid_concepts.csv',
        'test_data_path':      '/media/SSD_2TB/imageclef2025/splits/processed/dev_concepts.csv',
        'skip_head':           True,
        'split_token':         ',',
        'img_size':            (224, 224, 3),
    },
    'model': {
        # Replace B0 with V2-S here:
        'backbone':   tf.keras.applications.EfficientNetB0(
                          weights='imagenet',
                          include_top=False
                      ),
        'preprocessor': tf.keras.applications.efficientnet,

    #     'backbone': tf.keras.applications.ConvNeXtTiny(
    #     weights='imagenet',
    #     include_top=False
    # ),
    # 'preprocessor': tf.keras.applications.convnext,
        'data_format':  'channels_last',
    },
    'model_parameters': {
        'pooling':           'gem',
        'repr_dropout':      0.,
        'mlp_hidden_layers': [],
        'mlp_dropout':       0.,
        'use_sam':           False,
    },
    'training_parameters': {
        'loss': {
            'name': 'bce'
        },
        'epochs':               20,
        'batch_size':           16,
        'learning_rate':        1e-3,
        'patience_early_stopping': 3,
        'patience_reduce_lr':      1,
        # 'load_checkpoint': '/home/annachatz/med-tagger/Checkpoints/Monte Carlo/5_efficientnetb0_keras_2025_foivos.keras',
        # 'load_checkpoint':         '/home/annachatz/med-tagger/Checkpoints/efficientnetb0_2025_all.h5',
        # 'load_checkpoint':          '/home/annachatz/med-tagger/Checkpoints/efficientnetb0_trained_train_val.keras',
        # 'load_checkpoint':         '/home/annachatz/med-tagger/Checkpoints/Monte Carlo/5_efficientnetb0_keras_2025_foivos.keras',
        # 'skip_training':           True,
        'checkpoint_path':         '/home/annachatz/med-tagger/Checkpoints/efficientnetb0_curriculum.keras',
        # 'checkpoint_path': None
    },
    'save_results': True,
    'results_path': '/home/annachatz/med-tagger/Results/official_test_threshold_per_label_2025.csv'
}

t = TagCXN(configuration=configuration)
t.run()
# t.load_and_predict(off_test=True)