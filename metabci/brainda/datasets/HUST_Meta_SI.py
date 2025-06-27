# -*- coding: utf-8 -*-

import mne
import numpy as np
from mne.io import Raw, RawArray
from mne.channels import make_standard_montage
from .base import BaseDataset
from typing import Union, Optional, Dict, List, cast
from pathlib import Path
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

ROOT = 'D:/Speech_MetaBCI/'

class HUSTMetaSI(BaseDataset):
    
    _EVENTS = {
        "forward":  (1, (4, 5)),
        "backward": (2, (4, 5)),
        "left":     (3, (4, 5)),
        "right":    (4, (4, 5)),
    }

    _CHANNELS = [
        'EEG P3-Pz', 
        'EEG C3-Pz', 
        'EEG F3-Pz', 
        'EEG Fz-Pz', 
        'EEG F4-Pz', 
        'EEG C4-Pz', 
        'EEG P4-Pz', 
        'EEG Cz-Pz', 
        'EEG CM-Pz', 
        'EEG A1-Pz', 
        'EEG Fp1-Pz', 
        'EEG Fp2-Pz', 
        'EEG T3-Pz', 
        'EEG T5-Pz', 
        'EEG O1-Pz', 
        'EEG O2-Pz', 
        'EEG X3-Pz', 
        'EEG X2-Pz', 
        'EEG F7-Pz', 
        'EEG F8-Pz', 
        'EEG X1-Pz', 
        'EEG A2-Pz', 
        'EEG T6-Pz', 
        'EEG T4-Pz', 
        'Trigger'
    ]

    def __init__(self):
        super().__init__(
            dataset_code="HUST_Meta_SI",
            subjects=list(range(1, 2)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=300,
            paradigm="speech_imagery",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        subject = cast(int, subject)
        base_root = "{:s}Subject{:02d}{:s}".format(ROOT, subject, "/processed/")

        file_dest = mne_data_path(
            "{:s}{:s}.mat".format(base_root, 'eeg'),
            "HUST_Meta_SI",
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )
        dests = [[file_dest]]

        return dests

    # def _get_single_subject_data(
    #     self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    # ) -> Dict[str, Dict[str, Raw]]:
    #     dests = self.data_path(subject)
    #     montage = make_standard_montage("standard_1020")
    #     montage.rename_channels(
    #         {ch_name: ch_name.upper() for ch_name in montage.ch_names}
    #     )

    #     sess_arrays = np.append(loadmat(dests[0][0])["data"], loadmat(dests[1][0])["data"])

    #     sess = dict()
    #     for isess, sess_array in enumerate(sess_arrays):
    #         runs = dict()
    #         X = (sess_array.X).T * 1e-6  # volt
    #         trial = sess_array.trial
    #         y = sess_array.y
    #         stim = np.zeros((1, X.shape[-1]))

    #         if y.size > 0:
    #             stim[0, trial - 1] = y

    #         data = np.concatenate((X, stim), axis=0)

    #         ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + [
    #             "EOG1",
    #             "EOG2",
    #             "EOG3",
    #         ]
    #         ch_types = ["eeg"] * len(self._CHANNELS) + ["eog"] * 3
    #         ch_names = ch_names + ["STI 014"]
    #         ch_types = ch_types + ["stim"]

    #         info = mne.create_info(ch_names, self.srate, ch_types=ch_types)
    #         raw = RawArray(data, info)
    #         raw = upper_ch_names(raw)
    #         raw.set_montage(montage)
    #         runs["run_0"] = raw
    #         sess["session_{:d}".format(isess)] = runs
    #     return sess