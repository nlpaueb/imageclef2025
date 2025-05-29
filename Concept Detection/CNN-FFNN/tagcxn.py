### NORMAL
import gc
import json
import os

import numpy as np
import scipy as sp
from tqdm import tqdm

import utilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf


# ---------------------------------------------------------------------------
#  helper – build multi-hot matrix once (sparse-friendly)
# ---------------------------------------------------------------------------
def build_multihot_matrix(data_dict, label_to_idx):
    """
    data_dict maps image_id -> "tag1;tag2;..."
    returns (N, C) uint8 NumPy array
    """
    n = len(data_dict)
    c = len(label_to_idx)
    Y = np.zeros((n, c), dtype=np.uint8)
    for r, img in enumerate(data_dict):
        for tag in data_dict[img].split(';'):
            idx = label_to_idx.get(tag)
            if idx is not None:
                Y[r, idx] = 1
    return Y


# ---------------------------------------------------------------------------
#  helper – per-label F1
# ---------------------------------------------------------------------------
def label_f1_scores(y_true, y_pred_bin):
    """
    y_true, y_pred_bin : (N, C) {0,1}
    returns 1-D array of length C containing F1_c
    """
    tp = (y_true & y_pred_bin).sum(0).astype(np.float32)
    fp = (~y_true & y_pred_bin).sum(0).astype(np.float32)
    fn = (y_true & ~y_pred_bin).sum(0).astype(np.float32)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return f1


# ---------------------------------------------------------------------------
#  masked-BCE loss factory
# ---------------------------------------------------------------------------
def make_masked_bce(active_mask_var):
    """
    Returns a BinaryCrossentropy loss that:
      • multiplies y_true and y_pred by the current active_mask (length C)
      • ignores samples where *none* of their active labels are 1
    """
    # bce = tf.keras.losses.BinaryCrossentropy(reduction=None)
    # bce = tf.keras.losses.binary_crossentropy
    bce = tf.keras.ops.binary_crossentropy

    
    def loss_fn(y_true, y_pred):
        mask = tf.cast(active_mask_var, tf.float32)          # (C,)
        # mask = tf.expand_dims(mask, 0)                       # (1,C) broadcast
        y_true_masked = y_true * mask
        y_pred_masked = y_pred * mask

        per_label = bce(y_true_masked, y_pred_masked)        # (B, C)
        # per_sample = bce(y_true_masked, y_pred_masked)        # (B,)

        # per_label = -(
        #         y_true_masked * tf.math.log(y_pred_masked) +
        #         (1. - y_true_masked) * tf.math.log(1. - y_pred_masked)
        #     )         # (B,C)

        # zero-out inactive label positions
        per_label *= mask

        # remove samples that have zero active positives
        sample_weight = tf.cast(
            tf.reduce_sum(y_true_masked, axis=1) > 0, tf.float32
        )                                                    # (B,)
        loss_per_sample = tf.reduce_sum(per_label, axis=-1)   # (B,)
        # loss = tf.reduce_sum(loss_per_sample) / (
        #     tf.reduce_sum(sample_weight) + 1e-6
        # )

        # loss = loss_per_sample / (tf.reduce_sum(mask) + 1e-6)
        loss = loss_per_sample * sample_weight   # sample weight is 0 or 1 
        loss /= (tf.reduce_sum(mask) + 1e-6)

        return loss

    return loss_fn


def masked_binary_crossentropy(mask_indices, 
                               num_labels, 
                               from_logits: bool = False):
    """
    Returns a loss function which computes the binary crossentropy
    for each of the `num_labels` outputs but ignores (masks out)
    the labels at positions `mask_indices`.

    Args:
      mask_indices: list of int
        Indices in [0, num_labels) whose BCE contribution will be zeroed.
      num_labels: int
        Total number of labels in the multi-hot target vector.
      from_logits: bool
        Passed to tf.keras.losses.binary_crossentropy.

    Returns:
      A callable loss(y_true, y_pred) suitable for model.compile(...)
    """
    # build a 0/1 mask vector once
    mask = np.ones((num_labels,), dtype=np.float32)
    mask[mask_indices] = 0.0
    mask_tensor = tf.constant(mask)             # shape: (num_labels,)

    # precompute how many labels remain unmasked
    unmasked_count = tf.reduce_sum(mask_tensor) # scalar > 0

    
    def loss_fn(y_true, y_pred):
        # y_true, y_pred: shape (batch_size, num_labels)
        # computes per-label BCE, shape (batch_size, num_labels)
        bce = tf.keras.ops.binary_crossentropy(
            y_true, y_pred, from_logits=from_logits
        )

        # zero-out the masked labels
        masked_bce = bce * mask_tensor  # broadcast to (batch_size, num_labels)

        # sum over labels, then normalize by number of active labels
        per_sample_loss = tf.reduce_sum(masked_bce, axis=-1) / (unmasked_count + 1e-8)

        # returns shape (batch_size,), so Keras will then take the mean over batch
        return per_sample_loss

    return loss_fn



# ---------------------------------------------------------------------------
#  callback – updates the mask after warm-up
# ---------------------------------------------------------------------------
class CurriculumScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 active_mask_var,
                 sorted_indices,
                 stage_width,
                 tau,
                 warmup_epochs):
        super().__init__()
        self.m = active_mask_var
        self.sorted = sorted_indices          # list of label indices
        self.stage_w = stage_width
        self.tau = tau
        self.warm = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # epoch numbering starts at 0
        if epoch < self.warm:
            return

        stage = ((epoch - self.warm) // self.tau) + 1
        k = int(min(stage * self.stage_w, len(self.sorted)))
        active = np.zeros_like(self.m.numpy())
        active[self.sorted[:k]] = 1.          # 1.0 for active labels
        self.m.assign(active)
        tf.print(
            f"[Curriculum] Epoch {epoch+1}: stage {stage} – "
            f"activating {k}/{len(self.sorted)} labels"
        )


def topk_cooc(label_sets, C, K=50):
    
    """
    One pass over an iterable of *label_sets* (each a Python `set[int]`).

    Returns:
        dict{label -> list[(neigh, count)]}   (len ≤ K for every label)
    """
    import heapq
    heaps = [ [] for _ in range(C) ]              # min-heap per label
    seen   = [ set() for _ in range(C) ]          # track neighbours already enqueued

    for L in label_sets:
        L = list(L)
        for i, c in enumerate(L):
            for j in L[i+1:]:
                for a, b in ((c, j), (j, c)):     # directed update
                    h, s = heaps[a], seen[a]
                    if b in s:                    # already in heap → update count
                        # find & update entry
                        for k, (cnt, nbr) in enumerate(h):
                            if nbr == b:
                                h[k] = (cnt + 1, nbr)
                                heapq.heapify(h)
                                break
                    elif len(h) < K:              # heap not full
                        heapq.heappush(h, (1, b))
                        s.add(b)
                    else:                         # heap full → maybe replace min
                        cnt_min, b_min = h[0]
                        if 1 > cnt_min:           # every new pair starts at 1
                            heapq.heapreplace(h, (1, b))
                            s.discard(b_min); s.add(b)
    # convert heaps to sorted lists
    topk = {c: sorted([(nbr, cnt) for cnt, nbr in h],
                      key=lambda x: -x[1])
            for c, h in enumerate(heaps) if h}
    return topk


def adjusted_difficulty(s_c, topk_dict, alpha=0.3):
    """
    s_c        – NumPy 1-D array of raw difficulty scores (shape C)
    topk_dict  – output of `topk_cooc`
    alpha      – blending coefficient ∈ [0,1]

    returns s_adj (same shape as s_c)
    """
    s_adj = np.copy(s_c)
    for c, neighs in topk_dict.items():
        counts  = np.array([cnt  for _, cnt in neighs], dtype=np.float32)
        indices = np.array([nbr for nbr, _ in neighs], dtype=np.int32)
        denom   = counts.sum()
        if denom > 0:
            num = (counts * s_c[indices]).sum()
            corr = num / denom                  # weighted mean neighbour difficulty
            s_adj[c] = s_c[c] * (1. - alpha * corr)
    # return np.clip(s_adj, a_min=0.,)              # guard against negatives
    s_adj = s_adj.clip(0)
    return s_adj


def adjusted_difficulty_idf(s, A, temperature, alpha, beta, idf, ):  
    # Step 1: Apply rarity adjustment
    s_rarity = s * (1 + beta * idf)  
    
    # Step 2: Compute co-occurrence weights  
    W = sp.special.softmax(A / temperature, axis=1)
    # Step 3: Co-occurrence adjustment  
    # alpha_dynamic = alpha * (epoch / total_epochs)  
    weighted_s = W @ s_rarity
    term = alpha * weighted_s
    adjusted_s = s_rarity * (1 - term)  
    return adjusted_s  


class TagCXN:  # base model class

    def __init__(self, configuration):
        self.configuration = configuration
        self.backbone = self.configuration['model']['backbone'] 
        self.preprocessor = self.configuration['model']['preprocessor']
        self.train_images_folder = self.configuration['data']['train_images_folder']
        self.val_images_folder = self.configuration['data']['val_images_folder']
        self.test_images_folder = self.configuration['data']['test_images_folder']
        self.train_data_path = self.configuration['data']['train_data_path']
        self.val_data_path = self.configuration['data']['val_data_path']
        self.test_data_path = self.configuration['data']['test_data_path']
        self.img_size = self.configuration['data']['img_size']
        self.train_data, self.val_data, self.test_data = dict(), dict(), dict()
        self.train_img_index, self.train_concepts_index = dict(), dict()
        self.val_img_index, self.val_concepts_index = dict(), dict()
        self.test_img_index = dict()
        self.tags_list = list()

        self.model = None

    def init_structures(self, skip_head=False, split_token='\t'):
        if '.csv' in self.train_data_path:
            self.train_data = self.load_csv_data(self.train_data_path, skip_head=skip_head, 
                                                 split_token=split_token)
        else:
            self.train_data = self.load_json_data(self.train_data_path)
        if '.csv' in self.val_data_path:
            self.val_data = self.load_csv_data(self.val_data_path, skip_head=skip_head,
                                               split_token=split_token)
        else:
            self.val_data = self.load_json_data(self.val_data_path)
        if '.csv' in self.test_data_path:
            self.test_data = self.load_csv_data(self.test_data_path, skip_head=skip_head, 
                                                split_token=split_token)
        else:
            self.test_data = self.load_json_data(self.test_data_path)

        print('Number of training examples:', len(self.train_data), 'Number of validation examples:',
              len(self.val_data), 'Number of testing examples:',
              len(self.test_data))

        self.train_img_index, self.train_concepts_index = utils.create_index(self.train_data)
        self.val_img_index, self.val_concepts_index = utils.create_index(self.val_data)
        self.test_img_index, _ = utils.create_index(self.test_data) 

        self.tags_list = self.load_tags(self.train_data)
        print('Number of categories:', len(self.tags_list))

    @staticmethod
    def load_csv_data(file_name, skip_head=False, split_token='\t'):
        """
        loads .csv file into a Python dictionary.
        :param file_name: the path to the file (string)
        :param skip_head: whether to skip the first row of the file (if there is a header) (boolean)
        :return: data dictionary (dict)
        """
        data = dict()
        with open(file_name, 'r') as f:
            if skip_head:
                next(f)
            for line in f:
                image = line.replace('\n', '').split(split_token)
                concepts = image[1].split(';')
                if image[0]:
                    data[str(image[0] + '.jpg')] = ';'.join(concepts)
        print('Data loaded from:', file_name)
        return data

    ## for test 
    # @staticmethod
    # def load_csv_data(file_name, skip_head=False, split_token='\t'):
    #     """
    #     loads .csv file into a Python dictionary.
    #     :param file_name: the path to the file (string)
    #     :param skip_head: whether to skip the first row of the file (if there is a header) (boolean)
    #     :param split_token: delimiter between columns (string)
    #     :return: data dictionary mapping "ImageName.jpg" -> "tag1;tag2;..."
    #     """
    #     data = {}
    #     with open(file_name, 'r') as f:
    #         if skip_head:
    #             next(f)
    #         for line in f:
    #             parts = line.strip().split(split_token)
    #             img_id = parts[0]
    #             # if there's a second column, split it into concepts, otherwise leave empty
    #             concepts = parts[1].split(';') if len(parts) > 1 and parts[1] else []
    #             if img_id:
    #                 data[f"{img_id}.jpg"] = ';'.join(concepts)
    #     print('Data loaded from:', file_name)
    #     return data


    @staticmethod
    def load_json_data(file_name):
        """
        loads the data of JSON format into a Python dictionary
        :param file_name: the path to the file (string)
        :return: data dictionary (dict)
        """
        print('Data loaded from:', file_name)
        og = json.load(open(file=file_name, mode='r'))
        data = dict()
        for img in og:
            if 'normal' in og[img] and len(og[img]) == 1:
                og[img].remove('normal')
            data[img] = og[img]
        return data

    @staticmethod
    def load_tags(training_data):
        """
        loads the tags list
        :param training_data: training dictionary
        :return: the tags list
        """
        # if not isinstance(tags, list):
        #     return [line.strip() for line in open(tags, 'r')]
        tags = []
        for img in training_data:
            if isinstance(training_data[img], str):
                tags.extend(training_data[img].split(';'))
            else:
                tags.extend(training_data[img])
        # tags = set(tags)
        return list(dict.fromkeys(tags))

    def build_model(self, pooling, repr_dropout=0., mlp_hidden_layers=None,
                    mlp_dropout=0., use_sam=False, batch_size=None, data_format='channels_last'):
        """
        builds the Keras model
        :param pooling: global pooling method (string)
        :param repr_dropout: whether to apply dropout to the encoder's representation (rate != 0) (float)
        :param mlp_hidden_layers: a list containing the
        number of units of the MLP head. Leave None for plain linear (list)
        :param mlp_dropout: whether to apply dropout to the MLPs layers (rate != 0) (float)
        :param use_sam: whether to use SAM optimization (boolean)
        :param batch_size: the batch size of training (int)
        :param data_format: whether the channels will be last
        :return: Keras model
        """
        if data_format == 'channels_first':
            inp = tf.keras.layers.Input(shape=self.img_size[::-1], name='input')
            x = self.backbone(self.backbone.input, training=False).last_hidden_state
        else:
            inp = tf.keras.layers.Input(shape=self.img_size, name='input')
            x = self.backbone(self.backbone.input, training=False)

        encoder = tf.keras.Model(inputs=self.backbone.input, outputs=x, name='backbone')
        z = encoder(inp)
        if pooling == 'avg':
            z = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool', data_format=data_format)(z)
        elif pooling == 'max':
            z = tf.keras.layers.GlobalMaxPooling2D(name='max_pool', data_format=data_format)(z)
        else:
            z = utils.GeM(name='gem_pool', data_format=data_format)(z)

        if repr_dropout != 0.:
            z = tf.keras.layers.Dropout(rate=repr_dropout, name='repr_dropout')(z)
        for i, units in enumerate(mlp_hidden_layers):
            z = tf.keras.layers.Dense(units=units, activation='relu', name=f'MLP-layer-{i}')(z)
            if mlp_dropout != 0.:
                z = tf.keras.layers.Dropout(rate=mlp_dropout, name=f'MLP-dropout-{i}')(z)

        z = tf.keras.layers.Dense(units=len(self.tags_list), activation='sigmoid', name='LR')(z)
        model = tf.keras.Model(inputs=inp, outputs=z, name='TagCXN')
        if use_sam:
            assert batch_size // 4 == 0  # this must be divided exactly due to tf.split in the implementation of SAM.
            model = tf.keras.models.experimental.SharpnessAwareMinimization(
                model=model, num_batch_splits=(batch_size // 4), name='TagCXN_w_SAM'
            )
        return model

    def train(self, train_parameters):
        """
        method that trains the model
        :param train_parameters: model and training hyperparameters
        :return: a Keras history object
        """
        batch_size = train_parameters.get('batch_size')
        self.model = self.build_model(pooling=train_parameters.get('pooling'),
                                      repr_dropout=train_parameters.get('repr_dropout'),
                                      mlp_hidden_layers=train_parameters.get('mlp_hidden_layers'),
                                      mlp_dropout=train_parameters.get('mlp_dropout'),
                                      use_sam=train_parameters.get('use_sam'), batch_size=batch_size,
                                      data_format=self.configuration['model']['data_format'])
        

        # Load checkpoint if specified
        if train_parameters.get('load_checkpoint') is not None:
            checkpoint_path = train_parameters.get('load_checkpoint')
            print(f"Loading model weights from: {checkpoint_path}")
            # self.model.load_weights(checkpoint_path)
            self.model =  tf.keras.models.load_model(checkpoint_path)
            print("Checkpoint loaded successfully.")

        # Skip training if specified
        if train_parameters.get('skip_training', False):
            print("Skipping training as 'skip_training' is set to True.")
            return None
        # loss = None
        if train_parameters.get('loss', {}).get('name') == 'bce':
            loss = tf.keras.losses.BinaryCrossentropy()
        elif train_parameters.get('loss', {}).get('name') == 'focal':
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True, alpha=train_parameters.get('loss', {}).get('focal_alpha'),
                gamma=train_parameters.get('loss', {}).get('focal_gamma')
            )
        elif train_parameters.get('loss', {}).get('name') == 'asl':
            loss = utils.AsymmetricLoss(
                gamma_neg=train_parameters.get('loss', {}).get('asl_gamma_neg'),
                gamma_pos=train_parameters.get('loss', {}).get('asl_gamma_pos'),
                clip=train_parameters.get('loss', {}).get('asl_clip')
            )
        else:
            loss = utils.loss_1mf1_by_bce

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=train_parameters.get('learning_rate')),
            loss=loss
        )

        early_stopping = utils.ReturnBestEarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       patience=train_parameters.get('patience_early_stopping'),
                                                       restore_best_weights=True, verbose=1)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1,
                                                         patience=train_parameters.get('patience_reduce_lr'))

        print('\nTraining...')
        history = self.model.fit(
            self.train_generator(list(self.train_img_index), batch_size, self.tags_list),
            steps_per_epoch=int(np.ceil(len(self.train_img_index) / batch_size)),
            validation_data=self.val_generator(list(self.val_img_index), batch_size, self.tags_list),
            validation_steps=int(np.ceil(len(self.val_img_index) / batch_size)),
            callbacks=[early_stopping, reduce_lr], verbose=1, epochs=train_parameters['epochs']
        )
        print('\nEnd of training...')

        if train_parameters.get('checkpoint_path', None) is not None:
            self.model.save(train_parameters.get('checkpoint_path'))

        gc.collect()

        return history
    
    def train_curriculum(self, train_parameters):
        # rho = 0.1
        # rho = 0.032
        rho = 0.13
        # rho = 0.18
        epochs_per_subset = 2
        epochs_warm_start = 5
        softmax_temp = 20
        top_k = 100
        batch_size = train_parameters.get('batch_size')
        self.model = self.build_model(pooling=train_parameters.get('pooling'),
                                      repr_dropout=train_parameters.get('repr_dropout'),
                                      mlp_hidden_layers=train_parameters.get('mlp_hidden_layers'),
                                      mlp_dropout=train_parameters.get('mlp_dropout'),
                                      use_sam=train_parameters.get('use_sam'), batch_size=batch_size,
                                      data_format=self.configuration['model']['data_format'])
        
        # Load checkpoint if specified
        if train_parameters.get('load_checkpoint') is not None:
            checkpoint_path = train_parameters.get('load_checkpoint')
            print(f"Loading model weights from: {checkpoint_path}")
            # self.model.load_weights(checkpoint_path)
            self.model =  tf.keras.models.load_model(checkpoint_path)
            print("Checkpoint loaded successfully.")

        # Skip training if specified
        if train_parameters.get('skip_training', False):
            print("Skipping training as 'skip_training' is set to True.")
            return None
        
        active_mask = tf.Variable(np.ones(len(self.tags_list), dtype=np.float32), trainable=False, name='active_mask')
        print(active_mask)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_parameters.get('learning_rate')),
            loss=make_masked_bce(active_mask)
            # loss=masked_binary_crossentropy(mask_indices=[], num_labels=len(self.tags_list), )
            # loss='bce'
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1,
                                                         patience=train_parameters.get('patience_reduce_lr'))

        # -------------------------------------------------------------------
        # 2) WARM-UP (plain training, mask = all-ones)
        # -------------------------------------------------------------------
        print(f"\n[Curriculum] Warm-up for {epochs_warm_start} epoch(s)…")
        self.model.fit(
            self.train_generator(list(self.train_img_index), batch_size, self.tags_list),
            epochs=epochs_warm_start,
            steps_per_epoch=int(np.ceil(len(self.train_img_index) / batch_size)),
            validation_data=self.val_generator(list(self.val_img_index), batch_size, self.tags_list),
            validation_steps=int(np.ceil(len(self.val_img_index) / batch_size)),
            verbose=1, callbacks=[reduce_lr]
        )

        print("[Curriculum] Computing per-label F1 from warm-up…")
        bs = list(utils.divisor_generator(len(self.train_img_index)))[1]

        train_preds = self.model.predict(
            self.test_generator(list(self.train_img_index),
                                self.train_img_index, bs),
            steps=int(np.ceil(len(self.train_img_index) / bs)),
            verbose=1
        )

        y_true_train = build_multihot_matrix(self.train_data,
                                             {l: i for i, l in enumerate(self.tags_list)})
        y_pred_train = (train_preds >= 0.5).astype(np.uint8)
        f1_c = label_f1_scores(y_true_train, y_pred_train)           # (C,)
        s_c = 1. - f1_c                                      # difficulty

        # -------------------------------------------------------------------
        # 3) Correlation-adjust the difficulty (softmax with temperature)
        # -------------------------------------------------------------------
        print("[Curriculum] Building correlation matrix...")
        # Y_train = build_multihot_matrix(self.train_data,
        #                                 {l: i for i, l in enumerate(self.tags_list)})
        # A = y_true_train.T @ y_true_train                               # (C,C) counts
        A = sp.sparse.csr_matrix(y_true_train).T.dot(sp.sparse.csr_matrix(y_true_train))
        A = A.toarray()
        # softmax row-wise with temperature
        # A_soft = np.exp(A / softmax_temp)
        # W = A_soft / (A_soft.sum(axis=1, keepdims=True) + 1e-12)
        W = sp.special.softmax(A / softmax_temp, axis=1)
        neigh_term  = W @ s_c                                  # (C,)
        s_adj = s_c * (1.0 - 0.5 * neigh_term)        # (C,)
        # label_counts =  np.sum(y_true_train, axis=0)
        # idf = np.log(len(y_true_train) / (1 + label_counts))  
        # s_adj = adjusted_difficulty_idf(s_c, A, softmax_temp, alpha=0.3, beta=0.1, idf=idf)

        # easy → hard = low → high score
        sorted_lbl_idx = np.argsort(s_adj)                    # ascending
        stage_width = np.ceil(rho * len(self.tags_list))

        # print(f"[Curriculum] Streaming label sets to build top-{top_k} co-occurrence…")

        # # generator that yields a *set* of label-indices per training sample
        # def _label_set_stream():
        #     tag_to_idx = {l: i for i, l in enumerate(self.tags_list)}
        #     for img_id in self.train_data:
        #         tags = self.train_data[img_id].split(';')
        #         yield {tag_to_idx[t] for t in tags if t in tag_to_idx}

        # topk_dict = topk_cooc(_label_set_stream(), len(self.tags_list), K=top_k)
        # # print((topk_dict))

        # print("[Curriculum] Computing adjusted difficulty scores (formula)…")
        # s_adj = adjusted_difficulty(s_c, topk_dict, )

        # sorted_lbl_idx = np.argsort(s_adj)    # easy → hard as before
        # stage_width    = np.ceil(rho * len(self.tags_list))

        # print(f"[Curriculum] Stage width (number of classes to add) = {stage_width} labels.")

        # -------------------------------------------------------------------
        # 4) Attach CurriculumScheduler callback
        # -------------------------------------------------------------------
        sched = CurriculumScheduler(active_mask_var=active_mask,
                                    sorted_indices=list(sorted_lbl_idx),
                                    stage_width=stage_width,
                                    tau=epochs_per_subset,
                                    warmup_epochs=epochs_warm_start)
        
        print("\n[Curriculum] Starting curriculum training...")
        history = self.model.fit(
            self.train_generator(list(self.train_img_index), batch_size, self.tags_list),
            steps_per_epoch=int(np.ceil(len(self.train_img_index) / batch_size)),
            validation_data=self.val_generator(list(self.val_img_index), batch_size, self.tags_list),
            validation_steps=int(np.ceil(len(self.val_img_index) / batch_size)),
            callbacks=[sched, reduce_lr],       
            verbose=1,
            initial_epoch=epochs_warm_start,
            epochs=train_parameters['epochs']
        )

        print("\n[Curriculum] Training finished.")

        if train_parameters.get('checkpoint_path', None) is not None:
            self.model.save(train_parameters.get('checkpoint_path'))
        
        gc.collect()

        return history

    
    ## for inference on dev data
    def load_and_predict(self, off_test=False):
        self.init_structures(skip_head=self.configuration['data']['skip_head'], 
                            split_token=self.configuration['data']['split_token'])
        
        val_dev_data = None
        val_dev_img_index = None
        if off_test:
            self.off_test_data = []
            with open('/media/SSD_2TB/imageclef2025/test_ids.csv', 'r') as f:
                next(f)
                for line in f:
                    self.off_test_data.append(str(line).split('\n', 1)[0])
            print('Number of test instances:', len(self.off_test_data))  # has .jpg

            val_dev_data = {**self.val_data, **self.test_data}
            print(f'Merged validation and development data size: {len(val_dev_data)}.')
            val_dev_img_index, _ = utils.create_index(val_dev_data)

        params = self.configuration['training_parameters']
        params.update(self.configuration['model_parameters'])

        checkpoint_path = params.get('load_checkpoint')
        # print(f"Loading weights from {checkpoint_path}")
        # self.model.load_weights(checkpoint_path)
        self.model = tf.keras.models.load_model(checkpoint_path, compile = False, custom_objects={'GeM': utils.GeM})
        print(self.model.summary())

        print("Generating validation predictions to re-tune threshold...")

        if off_test:
            # bs = list(utils.divisor_generator(len(val_dev_img_index)))[1]
            bs = 1
            val_preds = self.model.predict(
                self.test_generator(list(val_dev_img_index), val_dev_img_index, bs),
                steps=int(np.ceil(len(val_dev_img_index) / bs)),
                verbose=1
            )
            # best_threshold, _ = self.tune_threshold(val_preds, data=val_dev_data)
            best_threshold = 0.4
            thresholds, _ = self.tune_threshold_per_label(predictions=val_preds, 
                                                          data=val_dev_data, heuristic=False)
        else:
            bs = list(utils.divisor_generator(len(self.val_img_index)))[1]
            val_preds = self.model.predict(
                self.test_generator(list(self.val_img_index), self.val_img_index, bs),
                steps=int(np.ceil(len(self.val_img_index) / bs)),
                verbose=1
            )
            # best_threshold, _ = self.tune_threshold(val_preds, data=self.val_data)
            thresholds, _ = self.tune_threshold_per_label(predictions=val_preds, 
                                                          data=self.val_data, heuristic=False)
            # thresholds = [0.4] * len(self.tags_list)
            # thresholds = np.array(thresholds)

        print("Generating test predictions...")
        test_score = -10.
        if off_test:
            bs = list(utils.divisor_generator(len(self.off_test_data)))[0]
            off_test_img_index = dict(zip(range(len(self.off_test_data)), list(self.off_test_data)))
            test_preds = self.model.predict(
                self.test_generator(list(off_test_img_index), off_test_img_index, bs, t='test'),
                steps=int(np.ceil(len(off_test_img_index) / bs)),
                verbose=1
            )

            # test_results = self.predict_only(best_threshold=best_threshold, predictions=test_preds)
            test_results = self.predict_only_threshold_per_label(thresholds=thresholds, predictions=test_preds)
        else:
            bs = list(utils.divisor_generator(len(self.test_img_index)))[1]
            test_preds = self.model.predict(
                self.test_generator(list(self.test_img_index), self.test_img_index, bs, t='test'),
                steps=int(np.ceil(len(self.test_img_index) / bs)),
                verbose=1
            )

            # test_score, test_results = self.test(best_threshold=best_threshold, predictions=test_preds)
            test_score, test_results = self.test_per_label_threshold(thresholds=thresholds,
                                                                     predictions=test_preds)

        if self.configuration.get('save_results'):
            print("Saving test results...")
            with open(self.configuration.get('results_path'), 'w') as f:
                for k, v in test_results.items():
                    f.write(f"{k},{v}\n")
            print("Results saved.")

        print(f"Final Test F1: {test_score:.4f} (Threshold: {best_threshold})")
        return test_score, best_threshold


    def train_generator(self, ids, batch_size, train_tags):
        """
        generator for training data
        :param ids: indices for each training sample in a batch (list)
        :param batch_size: batch size (int)
        :param train_tags: list of tags
        :return: yields a batch of data
        """
        batch = list()
        while True:
            np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):  # if not in the end of the list
                    if len(batch) == batch_size:
                        yield utils.load_batch(batch, self.train_img_index, self.train_concepts_index,
                                               self.train_images_folder, train_tags,
                                               self.preprocessor, size=self.img_size)

                        batch *= 0
                else:
                    yield utils.load_batch(batch, self.train_img_index, self.train_concepts_index,
                                           self.train_images_folder, train_tags, self.preprocessor, size=self.img_size)
                    batch *= 0

    def val_generator(self, ids, batch_size, train_tags):
        """
        generator for validation data
        :param ids: indices for each validation sample in a batch (list)
        :param batch_size: batch size (int)
        :param train_tags: list of tags
        :return: yields a batch of data
        """
        batch = list()
        while True:
            np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):
                    if len(batch) == batch_size:
                        yield utils.load_batch(batch, self.val_img_index,
                                               self.val_concepts_index, self.val_images_folder,
                                               train_tags, self.preprocessor, size=self.img_size, )
                        batch *= 0
                else:
                    yield utils.load_batch(batch, self.val_img_index, self.val_concepts_index,
                                           self.val_images_folder, train_tags, self.preprocessor, size=self.img_size, )
                    batch *= 0

    def test_generator(self, ids, index, batch_size, t='val'):
        """
        generator for testing data
        :param ids: indices for each testing sample in a batch (list)
        :param index: data index (dict)
        :param batch_size: batch size (int)
        :param t: flag for validation or testing (string)
        :return:
        """
        batch = list()
        while True:
            # np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):
                    if len(batch) == batch_size:
                        yield utils.load_test_batch(batch, index, self.val_images_folder
                                                    if t == 'val' else self.test_images_folder,
                                                    self.preprocessor, size=self.img_size)
                        batch *= 0
                else:
                    yield utils.load_test_batch(batch, index, self.val_images_folder
                                                if t == 'val' else self.test_images_folder,
                                                self.preprocessor, size=self.img_size)
                    batch *= 0

    def train_tune_test(self):
        """
        core logic of the file --> train, tune and test
        :return: a test score float, the test results in a dictionary format and a textual summary
        """
        self.init_structures(skip_head=self.configuration['data']['skip_head'], 
                             split_token=self.configuration['data']['split_token'])

        train_parameters = self.configuration['training_parameters']
        train_parameters.update(self.configuration['model_parameters'])

        # training_history = self.train(train_parameters=train_parameters)
        training_history = self.train_curriculum(train_parameters=train_parameters)

        # bs = list(utils.divisor_generator(len(self.val_img_index)))[1]
        bs = 1
        val_predictions = self.model.predict(self.test_generator(list(self.val_img_index),
                                                                 self.val_img_index, bs),
                                             verbose=1,
                                             steps=int(np.ceil(len(self.val_img_index) / bs)))
        print(val_predictions.shape)
        best_threshold = -1.
        best_threshold, val_score = self.tune_threshold(predictions=val_predictions, data=self.val_data,
                                                        not_bce=False)
        # threshdols, val_score = self.tune_threshold_per_label(predictions=val_predictions, data=self.val_data, heuristic=True)
        del val_predictions
        bs = list(utils.divisor_generator(len(self.test_img_index)))[1]
        test_predictions = self.model.predict(self.test_generator(list(self.test_img_index),
                                                                  self.test_img_index, bs, t='test'),
                                              verbose=1,
                                              steps=int(np.ceil(len(self.test_img_index) / bs)))
        print(test_predictions.shape)
        test_score, test_results = self.test(best_threshold=best_threshold,
                                             predictions=test_predictions, )
        # test_score, test_results = self.test_per_label_threshold(thresholds=threshdols, 
        #                                                          predictions=test_predictions)
        del test_predictions

        s = ('Development score = ' + str(test_score) +
             ' with threshold = ' + str(best_threshold) + ' and validation score = ' + str(val_score))

        return test_score, test_results, best_threshold, s

    def run(self):
        """
        basic run method
        :return: a dictionary of checkpoint paths alongside with scores and thresholds
        """

        
        thresholds_map = {}
        # test_scores = list()
        # info = list()

        test_score, test_results, best_threshold, txt = self.train_tune_test()
        print(txt)


        # test_scores.append(test_score)
        if self.configuration['training_parameters']['checkpoint_path'] is not None:
            thresholds_map[self.configuration['training_parameters']['checkpoint_path']] = [best_threshold, test_score]
        # info.append(txt)
        # for i in range(len(info)):
        #     print(info[i])
        # s = 'Mean dev score was: ' + str(sum(test_scores) / len(test_scores)) + '\n\n\n'
        # print(s)
        # info *= 0
        # test_scores *= 0
        #
        if self.configuration.get('save_results'):
            print('\n\nSaving results...\n')
            with open(self.configuration.get('results_path'), 'w') as out_test:
                for result in test_results:
                    out_test.write(result + ',' + test_results[result] + '\n')
            print('Results saved!')

        # pickle.dump(thresholds_map, open(str(self.backbone_name) + '_map.pkl', 'wb'))
        # pickle.dump(thresholds_map, open('temp_map.pkl', 'wb'))
        return thresholds_map

    def tune_threshold(self, predictions, data, not_bce=False):
        """
        method that tunes the classification threshold
        :param predictions: array of validation predictions (NumPy array)
        :param not_bce: flag for not bce losses (boolean)
        :return: best threshold and best validation score
        """
        print('\nGot predictions for validation set of split')
        print('\nTuning threshold for split #{}...')
        # steps = 100
        init_thr = 0.1
        if not_bce:
            init_thr = 0.3
        f1_scores = {}
        recalls = {}
        precisions = {}
        print('Initial threshold:', init_thr)
        for i in tqdm(np.arange(init_thr, 1, .1)):
            threshold = i
            # print('Current checking threshold =', threshold)
            y_pred_val = {}
            for j in range(len(predictions)):
                predicted_tags = []

                # indices of elements that are above the threshold.
                for index in np.argwhere(predictions[j] >= threshold).flatten():
                    predicted_tags.append(self.tags_list[index])

                # string with ';' after each tag. Will be split in the f1 calculations.
                # print(len(predicted_tags))
                y_pred_val[list(data.keys())[j]] = ';'.join(predicted_tags)
                # print(y_pred_val)

            f1_scores[threshold], p, r, _ = utils.evaluate_f1(data, y_pred_val, test=True)
            recalls[threshold] = r
            precisions[threshold] = p

        # get key with max value.
        best_threshold = max(f1_scores, key=f1_scores.get)
        print('The best F1 score on validation data' +
              ' is ' + str(f1_scores[best_threshold]) +
              ' achieved with threshold = ' + str(best_threshold) + '\n')

        # print('Recall:', recalls[best_threshold], ' Precision:', precisions[best_threshold])
        return best_threshold, f1_scores[best_threshold]
    
    def tune_threshold_per_label(self, predictions, data, heuristic=False):
        print('\nGot predictions for validation set of split')
        print('\nTuning a threshold per label...')
        # f1_scores = {}
        # recalls = {}
        # precisions = {}
        y_pred_val = {}
        label_to_idx = {label: i for i, label in enumerate(self.tags_list)}
        if heuristic:
            lambda_ = 0.5
            counts = np.zeros(len(self.tags_list), dtype=np.int64)
            for tag_str in data.values():
                for tag in tag_str.split(";"):
                    idx = label_to_idx.get(tag)
                    if idx is not None:
                        counts[idx] += 1
            
            thresholds = 0.5 - lambda_ * np.log(len(data) / (counts + 1e-9))
            thresholds = np.clip(thresholds, 0.05, 0.95)
        else:
            y_true = np.zeros((len(data), len(self.tags_list)), dtype=np.int8)
            for row, img_id in enumerate(list(data)):
                tag_str = data[img_id]
                tags = tag_str.split(';')
                for tag in tags:
                    idx = label_to_idx.get(tag)
                    if idx is not None:
                        y_true[row, idx] = 1
            
            
            thresholds_init = np.full(len(self.tags_list), 0.4, dtype=np.float32)
            thresholds = thresholds_init.copy()
            
            max_passes = 1

            p = predictions >= thresholds
            tp = (p & y_true).sum(1).astype(np.int32)
            psz  = p.sum(1).astype(np.int32)
            tsz  = y_true.sum(1).astype(np.int32)

            def _samples_f1(tp, pred_sz): return (2.0 * tp / np.maximum(pred_sz + tsz, 1)).mean()
            baseline_f1 = _samples_f1(tp, psz)

            for sweep in tqdm(range(max_passes)):
                improved = False
                for label_idx in tqdm(range(len(self.tags_list))):
                    s_col = predictions[:, label_idx]
                    order = np.argsort(s_col)           # ascending
                    best_tau = thresholds[label_idx]
                    best_f1  = baseline_f1

                    tp_l   = tp.copy()
                    pred_l = psz.copy()
                    pred_col = p[:, label_idx].copy()

                    for idx in order[::-1]:
                        if pred_col[idx]: 
                            continue

                        pred_col[idx] = True
                        pred_l[idx]  += 1
                        if y_true[idx, label_idx]:
                            tp_l[idx] += 1

                        cur_f1 = _samples_f1(tp_l, pred_l)
                        if cur_f1 > best_f1 + 1e-8:
                            best_f1  = cur_f1
                            best_tau = s_col[idx]
                    
                    if best_f1 > baseline_f1 + 1e-8:
                        mask_new = (predictions[:, label_idx] >= best_tau) & ~p[:, label_idx]
                        p[mask_new, label_idx] = True
                        psz[mask_new] += 1
                        tp[mask_new] += y_true[mask_new, label_idx]
                        thresholds[label_idx] = best_tau
                        baseline_f1 = best_f1
                        improved = True
                
                if not improved: 
                    print(f'Breaking at sweep: {sweep+1} due to no improvement!')
                    break

        for j in range(len(predictions)):
            predicted_tags = []
            # indices of elements that are above the threshold.
            for index in np.argwhere(predictions[j] >= thresholds).flatten():
                predicted_tags.append(self.tags_list[index])
            # string with ';' after each tag. Will be split in the f1 calculations.
            y_pred_val[list(data.keys())[j]] = ';'.join(predicted_tags)
            
        f1_score, p, r, _ = utils.evaluate_f1(data, y_pred_val, test=True)
        print('The best F1 score on validation data' +
              ' is ' + str(f1_score) + '\n')
        # print('Recall:', recalls[best_threshold], ' Precision:', precisions[best_threshold])
        return thresholds, f1_score


    def test(self, best_threshold, predictions):
        """
        method that performs the evaluation on the test data
        :param best_threshold: the tuned classification threshold (float)
        :param predictions: array of test predictions (NumPy array)
        :return: test score and test results dictionary
        """
        print('\nStarting evaluation on test set...')
        y_pred_test = dict()

        for i in tqdm(range(len(predictions))):
            predicted_tags = list()
            # bt = best_threshold
            for j in range(len(self.tags_list)):
                if predictions[i, j] >= best_threshold:
                    predicted_tags.append(str(self.tags_list[j]))

            # string! --> will be split in the f1 function

            # final_tags = list(set(set(predicted_tags).union(set(most_frequent_tags))))
            # temp = ';'.join(final_tags)
            temp = ';'.join(predicted_tags)
            y_pred_test[list(self.test_data)[i]] = temp

        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on the test set is: {}\n'.format(f1_score))
        # print('Precision score:', p)
        # print('Recall score:\n', r)
        # pickle.dump(y_pred_test, open(f'my_test_results_split_{split}.pkl', 'wb'))
        return f1_score, y_pred_test
    
    def test_per_label_threshold(self, thresholds, predictions):
        print('\nStarting evaluation on test set...')
        y_pred_test = {}

        for i in tqdm(range(len(predictions))):
            predicted_tags = []
            for j in range(len(self.tags_list)):
                if predictions[i, j] >= thresholds[j]:
                    predicted_tags.append(str(self.tags_list[j]))
            
            temp = ';'.join(predicted_tags)
            y_pred_test[list(self.test_data)[i]] = temp
        
        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on the test set is: {}\n'.format(f1_score))
        return f1_score, y_pred_test
    
    def predict_only(self, best_threshold, predictions):
        """
        method that performs the evaluation on test data without gold labels
        :param best_threshold: the tuned classification threshold (float)
        :param predictions: array of test predictions (NumPy array)
        :return: test results dictionary
        """
        print('\nStarting evaluation on test set...')
        y_pred_test = dict()

        for i in tqdm(range(len(predictions))):
            predicted_tags = list()
            # bt = best_threshold
            for j in range(len(self.tags_list)):
                if predictions[i, j] >= best_threshold:
                    predicted_tags.append(str(self.tags_list[j]))

            # string! --> will be split in the f1 function

            # final_tags = list(set(set(predicted_tags).union(set(most_frequent_tags))))
            # temp = ';'.join(final_tags)
            temp = ';'.join(predicted_tags)
            y_pred_test[list(self.off_test_data)[i]] = temp
        
        return y_pred_test
    
    def predict_only_threshold_per_label(self, thresholds, predictions):
        """
        method that performs the evaluation on test data without gold labels
        :param best_threshold: the tuned classification threshold (float)
        :param predictions: array of test predictions (NumPy array)
        :return: test results dictionary
        """
        print('\nStarting evaluation on test set...')
        y_pred_test = dict()

        for i in tqdm(range(len(predictions))):
            predicted_tags = list()
            # bt = best_threshold
            for j in range(len(self.tags_list)):
                if predictions[i, j] >= thresholds[j]:
                    predicted_tags.append(str(self.tags_list[j]))

            # string! --> will be split in the f1 function

            # final_tags = list(set(set(predicted_tags).union(set(most_frequent_tags))))
            # temp = ';'.join(final_tags)
            temp = ';'.join(predicted_tags)
            y_pred_test[list(self.off_test_data)[i]] = temp
        
        return y_pred_test