import os
import json
from typing import Union, Tuple
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, math_img
from .utils import get_parcels_path, get_parcels_labels_path, ParcelsConfig
from collections import namedtuple
import numpy as np


def get_parcels(
    parcels: Union[str, ParcelsConfig]
) -> Tuple[Nifti1Image, dict]:
    """
    Get parcels image and labels.
    """
    if isinstance(parcels, str):
        parcels_img, label_dict = _get_saved_parcels(parcels)
        if parcels_img is None:
            parcels_img, label_dict = _get_external_parcels(
                ParcelsConfig(parcels_path=parcels)
            )
    else:
        parcels_img, label_dict = _get_external_parcels(parcels)

    return parcels_img, label_dict


def _get_saved_parcels(parcels: str) -> Tuple[Nifti1Image, dict]:
    """
    Get parcels image and labels from a saved parcels file.

    Parameters
    ----------
    parcels : str
        Parcels name.

    Returns
    -------
    parcels_img : Nifti1Image
        Parcels image.
    label_dict : dict
        Labels dictionary, mapping the numerical labels in Nifti1Image to
        label names.
    """
    parcels_path = get_parcels_path(parcels)
    if parcels_path is None or not os.path.exists(parcels_path):
        return None, None
    parcels_img = load_img(parcels_path)
    parcels_img = math_img("np.round(img)", img=parcels_img)

    parcels_labels_path = get_parcels_labels_path(parcels)
    if parcels_labels_path is not None and os.path.exists(parcels_labels_path):
        label_dict = json.load(open(parcels_labels_path))
        label_dict = {int(k): v for k, v in label_dict.items()}
    else:
        label_dict = {}
        for label in np.unique(parcels_img.get_fdata()):
            if label != 0:
                label_dict[int(label)] = int(label)
    return parcels_img, label_dict


def _get_external_parcels(parcels: ParcelsConfig) -> Tuple[Nifti1Image, dict]:
    """
    Get parcels image and labels from an external parcels file.

    Parameters
    ----------
    parcels : ParcelsConfig
        Parcels configuration.

    Returns
    -------
    parcels_img : Nifti1Image
        Parcels image.
    label_dict : dict
        Labels dictionary, mapping the numerical labels in Nifti1Image to
        label names.
    """
    if parcels.parcels_path is None or not os.path.exists(
        parcels.parcels_path
    ):
        return None, None

    parcels_img = load_img(parcels.parcels_path)
    parcels_img = math_img("np.round(img)", img=parcels_img)

    if parcels.labels_path is not None and os.path.exists(parcels.labels_path):
        label_dict = json.load(open(parcels.labels_path))
        label_dict = {int(k): v for k, v in label_dict.items()}
    else:
        label_dict = {}
        for label in np.unique(parcels_img.get_fdata()):
            if label != 0:
                label_dict[int(label)] = int(label)
    return parcels_img, label_dict
