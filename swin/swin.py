import shutil
import os
import json
import multiprocessing

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from timm import create_model
from fastai.vision.all import set_seed
from fastai.vision.all import torch, F
from fastai.vision.all import Path, ImageDataLoaders, RegressionBlock
from fastai.vision.all import setup_aug_tfms, Resize, Brightness, Contrast, Hue, Saturation, Warp, Zoom
from fastai.vision.all import Learner, SaveModelCallback, EarlyStoppingCallback


def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))


class PetfinderTransformer(object):
    def __init__(self, seed, dataset_parent=None, train_test_json_path=None, batch_size=None,
                 save_path=None, num_workers=-1, model_name=None):

        set_seed(seed, reproducible=True)
        self.split = None
        self.val_split_pct = None
        self.test_ids = None
        self.image_size = None
        self.dls = None
        self.learn = None
        self.optimal_lr = None
        self.epochs = None
        self.test_df = None
        self.train_df = None
        self.rmse = None
        self.loss_function = None
        self.dataset_path = None
        self.model = None
        self.model_name = model_name
        self.batch_size = batch_size
        self.save_path = save_path
        self.num_workers = num_workers
        self.seed = seed
        self.train_test_json_path = train_test_json_path
        self.dataset_parent = dataset_parent

    def prepare_test_dataset(self, dataset_parent=None, train_test_json_path=None):
        if dataset_parent is None:
            dataset_parent = self.dataset_parent
        if train_test_json_path is None:
            train_test_json_path = self.train_test_json_path

        self.test_ids = []
        if not os.path.exists(train_test_json_path):
            raise ValueError(f"File not found: {train_test_json_path}")
        self.split = json.load(open(train_test_json_path, "r"))
        test_filenames = self.split["test"]
        for test_filename in tqdm(test_filenames):
            test_filename = os.path.basename(test_filename)
            self.test_ids.append(test_filename.replace(".jpg", ""))
            if not os.path.exists(os.path.join(dataset_parent, f"test/{test_filename}")):
                shutil.move(os.path.join(dataset_parent, f"train/{test_filename}"),
                            os.path.join(dataset_parent, f"test/{test_filename}"))
        self.val_split_pct = len(self.split["val"]) / (len(self.split["val"]) + len(self.split["train"]))

    def prepare_train_val_dataset(self, dataset_parent=None):
        if dataset_parent is None:
            dataset_parent = self.dataset_parent
        self.dataset_path = Path(dataset_parent)
        train_df = pd.read_csv(self.dataset_path / 'train.csv')
        train_df = train_df[~train_df["Id"].isin(self.test_ids)]
        train_df['path'] = train_df['Id'].map(lambda x: str(self.dataset_path / 'train' / x) + '.jpg')
        train_df = train_df.drop(columns=['Id'])
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_df['pawpularity_float'] = train_df['Pawpularity'] / 100
        self.train_df = train_df

        # test data
        test_df = pd.read_csv(self.dataset_path / 'train.csv')
        test_df = test_df[test_df["Id"].isin(self.test_ids)]
        test_df['path'] = test_df['Id'].map(lambda x: str(self.dataset_path / 'test' / x) + '.jpg')
        test_df.drop(columns=['Id'], inplace=True)
        test_df['pawpularity_float'] = test_df['Pawpularity'] / 100
        self.test_df = test_df
        return train_df, test_df

    def prepare_dls(self, train_df=None, val_split_pct=None,
                    batch_size=None, num_workers=None, image_size=None):
        if train_df is None:
            train_df = self.train_df
        if val_split_pct is None:
            val_split_pct = self.val_split_pct
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        if os.name == 'nt':
            num_workers = 0
        if image_size is None:
            image_size = self.image_size
        self.dls = ImageDataLoaders.from_df(train_df,
                                            valid_pct=val_split_pct,
                                            seed=self.seed,
                                            fn_col='path',
                                            label_col='pawpularity_float',
                                            y_block=RegressionBlock,
                                            bs=batch_size,
                                            num_workers=num_workers,
                                            item_tfms=Resize(image_size),
                                            batch_tfms=setup_aug_tfms([Brightness(),
                                                                       Contrast(), Hue(),
                                                                       Saturation(), Warp(),
                                                                       Zoom()]))

        return self.dls

    def vizualize_batch(self, dls=None):
        if dls is None:
            dls = self.dls
        dls.show_batch()

    def get_model(self, dls=None, model_name=None, pretrained=True):
        if dls is None:
            dls = self.dls
        if model_name is None:
            model_name = self.model_name

        self.model = create_model(model_name, pretrained=pretrained, num_classes=dls.c)
        return self.model

    def build_learner(self, dls=None, model=None, loss_function=None):
        if dls is None:
            dls = self.dls
        if loss_function is None:
            loss_function = self.loss_function
        if model is None:
            model = self.model
        self.learn = Learner(dls, model, loss_func=loss_function, metrics=petfinder_rmse).to_fp16()
        self.optimal_lr = self.learn.lr_find(end_lr=3e-2)
        return self.learn, self.optimal_lr

    def fit_learner(self, learn=None, epochs=None, optimal_lr=None, save_path=None):
        if epochs is None:
            epochs = self.epochs
        if optimal_lr is None:
            optimal_lr = self.optimal_lr
        if learn is None:
            learn = self.learn
        if save_path is None:
            save_path = self.save_path
        learn.fit_one_cycle(epochs, float(optimal_lr.valley), cbs=[SaveModelCallback(),
                                                                   EarlyStoppingCallback(monitor='petfinder_rmse',
                                                                                         comp=np.less, patience=3)])
        learn.recorder.plot_loss()
        learn = learn.to_fp32()
        if save_path is not None:
            learn.save(save_path)
            learn.export(save_path)
        self.learn = learn
        return learn

    def evaluate_model(self, learn=None, dls=None, test_df=None):
        if learn is None:
            learn = self.learn
        if dls is None:
            dls = self.dls
        if test_df is None:
            test_df = self.test_df
        test_dl = dls.test_dl(test_df)
        preds, _ = learn.tta(dl=test_dl, n=5, beta=0)
        self.rmse = mean_squared_error(preds, test_df["pawpularity_float"].values, squared=False)
        return self.rmse

    def wrap_model(self, dataset_parent=None, train_test_json_path=None,
                   batch_size=None, num_workers=-1, image_size=None, model_name=None, epochs=None):
        # call this function to train and evaluate model
        if dataset_parent is None and self.dataset_parent is None:
            raise ValueError("Please provide a dataset_parent")
        if train_test_json_path is None and self.train_test_json_path is None:
            raise ValueError("Please provide a train_test_json_path")
        self.dataset_parent = dataset_parent
        self.train_test_json_path = train_test_json_path

        # now start with prep and training
        self.prepare_test_dataset()
        self.prepare_train_val_dataset()

        self.prepare_dls(batch_size=batch_size, num_workers=num_workers, image_size=image_size)
        self.vizualize_batch()
        self.get_model(model_name=model_name)
        self.build_learner()
        self.fit_learner(epochs=epochs)
        rmse = self.evaluate_model()
        return rmse
