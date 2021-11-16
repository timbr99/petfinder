import shutil
import os
import json
import multiprocessing
from typing import Tuple, Any, Union

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from timm import create_model
from fastai.vision.all import set_seed
from fastai.vision.all import torch, F
from fastai.vision.all import Path, ImageDataLoaders, RegressionBlock
from fastai.vision.all import setup_aug_tfms, Resize, Brightness, Contrast, Hue, Saturation, Warp, Zoom
from fastai.vision.all import Learner, LRFinder, SaveModelCallback, EarlyStoppingCallback, CSVLogger
from fastai.vision.all import BCEWithLogitsLossFlat, BCELossFlat, MSELossFlat, L1LossFlat, LabelSmoothingCrossEntropy


def custom_rmse(input_score, target_score):
    """
    Custom root mean squared error calculation function utilizing sigmoid function.

    :param input_score: Model prediction tensor
    :param target_score: Actual score as float
    :return: RMSE * 100
    """
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input_score.flatten()), target_score))


class PetfinderTransformer(object):
    def __init__(self, seed: int, dataset_parent: str = None, train_test_json_path: str = None, batch_size: int = None,
                 save_path: str = None, num_workers: int = -1, model_name: str = None,
                 loss_function_dict: dict = None) -> None:
        """
        Initialize Petfinder Swin Transformer class.

        :param seed: Random seed passed to the fastai set_seed function
        :param dataset_parent: parent path of the dataset (under this dir there should be a folder for train and test files)
        :param train_test_json_path: path to a JSON with IDs of images used for testing the model (this is dirty but
                                     necessary to fairly compare our models)
        :param batch_size: Batch size for training (decrease for less memory use -> A100 fits 108)
        :param save_path: path where model should be saved
        :param num_workers: CPU cores to work on (if on Windows this will be forced to 0)
        :param model_name: Name of pretrained model that will be loaded from TIMM
        :param loss_function_dict: Dictionary with callable loss functions
        """

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
        self.learner_recorder = None
        self.model_name = model_name
        self.batch_size = batch_size
        self.save_path = save_path
        self.num_workers = num_workers
        self.seed = seed
        self.train_test_json_path = train_test_json_path
        self.dataset_parent = dataset_parent
        if loss_function_dict is None:
            self.loss_function_dict = self.default_loss_functions()
        else:
            self.loss_function_dict = loss_function_dict

    def default_loss_functions(self):
        loss_function_dict = {
            "BCEWithLogitsLossFlat": BCEWithLogitsLossFlat(),
            "BCELossFlat": BCELossFlat(),
            "MSELossFlat": MSELossFlat(),
            "L1LossFlat": L1LossFlat(),
            "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy()
        }
        return loss_function_dict

    def prepare_test_dataset(self, dataset_parent: str = None, train_test_json_path: str = None) -> None:
        """
        Prepare the test dataset by coping files from the train folder to the test folder according to the
        train_test_json.

        :param dataset_parent: parent path of the dataset (under this dir there should be a folder for train and test files)
        :param train_test_json_path: path to a JSON with IDs of images used for testing the model (this is dirty but
                                     necessary to fairly compare our models)
        :return: None
        """
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

    def prepare_train_val_dataset(self, dataset_parent: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare train and validation dataset by dropping all IDs in test dataset from train dataframe.
        Additionally prepare test dataframe.
        Convert integer Pawpularity score to float between 0 and 1.

        :param dataset_parent: parent path of the dataset (under this dir there should be a folder for train and test files)
        :return: train and test dataframe
        """
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

    def prepare_dls(self, train_df: pd.DataFrame = None, val_split_pct: float = None,
                    batch_size: int = None, num_workers: int = None, image_size: int = None) -> ImageDataLoaders:
        """
        Prepare ImageDataLoader by providing train dataframe, percentage of validation split, batch size, CPU count and
        image size. For reference see https://docs.fast.ai/vision.data.html#ImageDataLoaders.

        :param train_df: Dataframe with training data
        :param val_split_pct: Percentage of dataset used for validation of model after each epoch.
        :param batch_size: Batch size for training (decrease for less memory use -> A100 fits 108)
        :param num_workers: CPU count (-1 to use all cores determined by multiprocessing.cpu_count()). If os is Windows
                            this parameter will be forced to 0 as MS does not properly work with multiprocessing.
        :param image_size: Size of images (must be square). Must match up with input layer of pretrained model.
        :return: ImageDataLoader class object
        """
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

        # check if os is Windows
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        return self.dls

    def visualize_batch(self, dls: ImageDataLoaders = None) -> None:
        """
        Use built-in fastai method to visualize images from dataset.

        :param dls: ImageDataLoader class object (https://docs.fast.ai/vision.data.html#ImageDataLoaders)
        :return: None
        """
        if dls is None:
            dls = self.dls
        dls.show_batch()

    def get_model(self, dls: ImageDataLoaders = None, model_name: str = None, pretrained: bool = True) -> Any:
        """
        Pull model from TIMM and load it into model property. Internet access may be required if model is pulled for
        the first time. Please see https://github.com/rwightman/pytorch-image-models for further details.

        :param dls: ImageDataLoader class object (https://docs.fast.ai/vision.data.html#ImageDataLoaders)
        :param model_name: Name of model as referenced in TIMM
        :param pretrained: If true use a pretrained model instead of a freshly initialised model
        :return: Model object (this can be any object, depending on the model requested)
        """
        if dls is None:
            dls = self.dls
        if model_name is None:
            model_name = self.model_name

        self.model = create_model(model_name, pretrained=pretrained, num_classes=dls.c)
        return self.model

    def build_learner(self, dls: ImageDataLoaders = None, model: Any = None,
                      loss_function: Union[
                          BCEWithLogitsLossFlat, BCELossFlat, MSELossFlat,
                          L1LossFlat, LabelSmoothingCrossEntropy] = None) -> Tuple[Learner, LRFinder]:
        """
        Transfer the model to a fastai Learner object for transfer learning. Determine the optimal learning rate
        by using built-in lr_find (https://fastai1.fast.ai/callbacks.lr_finder.html) of Learner object.

        :param dls: ImageDataLoader class object (https://docs.fast.ai/vision.data.html#ImageDataLoaders)
        :param model: Model object (this can be any object, depending on the model requested)
        :param loss_function: Object that defines the loss function (https://docs.fast.ai/losses.html)
        :return: Learner object (https://docs.fast.ai/learner.html) and LRFinder object
        """
        if dls is None:
            dls = self.dls
        if loss_function is None:
            loss_function = self.loss_function
        if model is None:
            model = self.model
        self.learn = Learner(dls, model, loss_func=loss_function, metrics=custom_rmse).to_fp16()
        self.optimal_lr = self.learn.lr_find(end_lr=3e-2)
        return self.learn, self.optimal_lr

    def fit_learner(self, learn: Learner = None, epochs: int = None, optimal_lr: LRFinder = None,
                    save_path: str = None) -> Learner:
        """
        Fit the Learner object on training data. Early stopping after three epochs without improvement of custom_rmse
        metric.

        :param learn: Learner object (https://docs.fast.ai/learner.html)
        :param epochs: Number of training epochs
        :param optimal_lr: LRFinder object with valley property (https://fastai1.fast.ai/callbacks.lr_finder.html)
        :param save_path: Save path of the model.
        :return: Learner object (https://docs.fast.ai/learner.html)
        """
        if epochs is None:
            epochs = self.epochs
        if optimal_lr is None:
            optimal_lr = self.optimal_lr
        if learn is None:
            learn = self.learn
        if save_path is None:
            save_path = self.save_path
        learn.fit_one_cycle(epochs, float(optimal_lr.valley), cbs=[SaveModelCallback(),
                                                                   EarlyStoppingCallback(monitor='custom_rmse',
                                                                                         comp=np.less, patience=3),
                                                                   CSVLogger()])
        self.learner_recorder = learn.recorder.__dict__
        learn = learn.to_fp32()
        if save_path is not None:
            learn.save(save_path)
            learn.export(os.path.join(save_path, "export.pkl"))
        self.learn = learn
        self.epochs = epochs
        return learn

    def evaluate_model(self, learn: Learner = None, dls: ImageDataLoaders = None,
                       test_df: pd.DataFrame = None) -> float:
        """
        Evaluate the model by predicting on test data and returning RMSE.

        :param learn: Learner object (https://docs.fast.ai/learner.html)
        :param dls: ImageDataLoader class object (https://docs.fast.ai/vision.data.html#ImageDataLoaders)
        :param test_df: Dataframe with test data and labels
        :return: RMSE (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
        """
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

    def read_config(self, config_path: str) -> None:
        """
        Reads config from specified file path. Sets variables to recreate training.

        :param config_path: Path to config file
        :return: None
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find file: {config_path}")
        config = json.load(open(config_path, "r"))
        self.dataset_parent = config.get("dataset_parent")
        self.train_test_json_path = config.get("train_test_json_path")
        self.batch_size = config.get("batch_size")
        self.num_workers = config.get("num_workers")
        self.image_size = config.get("image_size")
        self.model_name = config.get("model_name")
        self.epochs = config.get("epochs")
        self.loss_function = self.get_loss_function(config.get("loss_function"))

    def write_config(self, save_path: str = None) -> None:
        """
        Write variables required for training to JSON file in save path to recreate training.

        :param save_path: Path where model is saved
        :return: None
        """
        if save_path is None:
            if self.save_path is None:
                raise ValueError("Cannot write config, as save_path is None.")
            save_path = self.save_path
        config = {
            "dataset_parent": self.dataset_parent,
            "train_test_json_path": self.train_test_json_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "image_size": self.image_size,
            "model_name": self.model_name,
            "epochs": self.epochs,
            "loss_function": type(self.loss_function).__name__
        }
        json.dump(config, open(os.path.join(save_path, "config.json"), "w"))

    def get_loss_function(self, loss_function: str = None) -> object:
        """
        Returns loss function from loss function library.

        :param loss_function: Name of desired loss function
        :return: Loss function
        """
        if loss_function is None:
            return BCEWithLogitsLossFlat()
        else:
            return self.loss_function_dict.get(loss_function)

    def wrap_model(self, config: str = None, save_config: bool = True, dataset_parent: str = None,
                   train_test_json_path: str = None, loss_function: Union[str, object] = None,
                   batch_size: int = None, num_workers: int = -1, image_size: int = None, model_name: str = None,
                   epochs: int = None, save_path: str = None) -> None:
        """
        This function wraps the entire training and evaluation process into one function. Call it to recreate training.

        :param config: Path to config file. This will override init parameters but subside to custom parameters passed
                       to this function.
        :param save_config: Save parameters used for training to JSON file
        :param dataset_parent: parent path of the dataset (under this dir there should be a folder for train and test files)
        :param train_test_json_path: path to a JSON with IDs of images used for testing the model (this is dirty but
                                     necessary to fairly compare our models)
        :param loss_function: Loss function as string or callable function
        :param batch_size: Batch size for training (decrease for less memory use -> A100 fits 108)
        :param num_workers: CPU cores to work on (if on Windows this will be forced to 0)
        :param image_size: Size of images (must be square). Must match up with input layer of pretrained model.
        :param model_name: Name of model as referenced in TIMM
        :param epochs: Number of training epochs
        :param save_path: Save path of the model.
        :return: None
        """
        if config is not None:
            self.read_config(config)

        # call this function to train and evaluate model
        if dataset_parent is None and self.dataset_parent is None:
            raise ValueError("Please provide a dataset_parent")
        if train_test_json_path is None and self.train_test_json_path is None:
            raise ValueError("Please provide a train_test_json_path")
        if save_path is not None:
            self.save_path = save_path
        if dataset_parent is not None:
            self.dataset_parent = dataset_parent
        if train_test_json_path is not None:
            self.train_test_json_path = train_test_json_path
        if loss_function is not None:
            if type(loss_function) == str:
                self.loss_function = self.get_loss_function(loss_function)
            else:
                self.loss_function = loss_function

        # now start with prep and training
        self.prepare_test_dataset()
        self.prepare_train_val_dataset()

        self.prepare_dls(batch_size=batch_size, num_workers=num_workers, image_size=image_size)
        self.visualize_batch()
        self.get_model(model_name=model_name)
        self.build_learner()
        self.fit_learner(epochs=epochs)
        print("Model successfully fit.")
        self.rmse = self.evaluate_model()
        print(f"Model RMSE: {self.rmse}.")

        if save_config:
            self.write_config()
            print(f"Config file saved to path {self.save_path}/config.json")
