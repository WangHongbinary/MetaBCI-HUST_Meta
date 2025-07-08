# MetaBCI-HUST\_Meta

MetaBCI-HUST\_Meta was forked from MetaBCI for the [World Robot Contest 2025: BCI-Controlled Robot Competition](https://www.worldrobotconference.com/contest/BCIBrainControlRobotCompetition/2025.html). This repository contains all the materials related to the "Speech Imagery-Based Brain-Controlled Intelligent Car" project developed by the HUST\_Meta team based on the MetaBCI platform.

## Development on `brainstim`

This project first implements the speech imagery paradigm stimulation and decoding result feedback. The entry point is located in the `"SI"` section of the `demos/brainstim_demos/stim_demo.py` file, and the core implementation resides in the `elif pdim == "si"` block in `metabci/brainstim/paradigm.py`. To test this functionality, please run `demos/brainstim_demos/stim_demo.py`.

## Development on `brainda`

The project implements seven neural network architectures that can be used for speech decoding BCIs. These are located in `metabci/brainda/algorithms/deep_learning`. The models include `tonal_Net.py, fast.py, eeg_deformer.py, duin_cls.py, eeg_conformer.py, brainmodule.py, cnn_gru.py`. We would like to express our sincere gratitude to [tonal_BCI](https://github.com/yuanningli/tonal_BCI_decoding/tree/publications), [FAST](https://github.com/Jiang-Muyun/FAST), [EEG-Deformer](https://github.com/yi-ding-cs/EEG-Deformer), [Du-IN](https://github.com/liulab-repository/Du-IN), [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), [brainmagick](https://github.com/facebookresearch/brainmagick) and other excellent open-source repositories for their invaluable support in making this implementation possible.

## Development on `brainflow`

This project also enables online control of the intelligent car. This part is mainly implemented in the `demos/brainflow_demos/Online_si.py` file. To test it, simply run this script.

**Note:** Testing the online control system via brainflow requires the following:

1. Connect hardware devices including the Neuracle DSI-24 EEG cap, the intelligent car, and the Neuracle trigger box.
2. Launch DSI-Streamer software (we used version 1.08.44), establish a connection with the EEG cap, activate the TCP/IP socket, and start data recording.
3. Run `demos/brainflow_demos/Online_si.py`, and wait until the terminal displays a successful connection. At this point, the online module is waiting for triggers to begin data collection and decoding.
4. Run `demos/brainstim_demos/stim_demo.py` to launch the paradigm stimulation program. In the MetaBCI paradigm selection interface, choose the "SI" paradigm to start stimulation. After each trial, the decoding result and control command status will be shown in the terminal running `Online_si.py`.

# Introduction to the MetaBCI Platform

The following section provides an overview of the MetaBCI platform, which serves as the foundation for the developments and extensions presented in this repository.

# MetaBCI

## Welcome! 
MetaBCI is an open-source platform for non-invasive brain computer interface. The project of MetaBCI is led by Prof. Minpeng Xu from Tianjin University, China. MetaBCI has 3 main parts:
* brainda: for importing dataset, pre-processing EEG data and implementing EEG decoding algorithms.
* brainflow: a high speed EEG online data processing framework.
* brainstim: a simple and efficient BCI experiment paradigms design module. 

This is the first release of MetaBCI, our team will continue to maintain the repository. If you need the handbook of this repository, please contact us by sending email to TBC_TJU_2022@163.com with the following information:
* Name of your teamleader
* Name of your university(or organization)

We will send you a copy of the handbook as soon as we receive your information.

## Paper

If you find MetaBCI useful in your research, please cite:

Mei, J., Luo, R., Xu, L., Zhao, W., Wen, S., Wang, K., ... & Ming, D. (2023). MetaBCI: An open-source platform for brain-computer interfaces. Computers in Biology and Medicine, 107806.

And this open access paper can be found here: [MetaBCI](https://www.sciencedirect.com/science/article/pii/S0010482523012714)

## Content

- [MetaBCI-HUST\_Meta](#metabci-hust_meta)
  - [Development on `brainstim`](#development-on-brainstim)
  - [Development on `brainda`](#development-on-brainda)
  - [Development on `brainflow`](#development-on-brainflow)
- [Introduction to the MetaBCI Platform](#introduction-to-the-metabci-platform)
- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [Paper](#paper)
  - [Content](#content)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [What do we need?](#what-do-we-need)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* BCI datasets come in different formats and standards
* It's tedious to figure out the details of the data
* Lack of python implementations of modern decoding algorithms
* It's not an easy thing to perform BCI experiments especially for the online ones.

If someone new to the BCI wants to do some interesting research, most of their time would be spent on preprocessing the data, reproducing the algorithm in the paper, and also find it difficult to bring the algorithms into BCI experiments.

### The solution

The Meta-BCI will:

* Allow users to load the data easily without knowing the details
* Provide flexible hook functions to control the preprocessing flow
* Provide the latest decoding algorithms
* Provide the experiment UI for different paradigms (e.g. MI, P300 and SSVEP)
* Provide the online data acquiring pipeline.
* Allow users to bring their pre-trained models to the online decoding pipeline.

The goal of the Meta-BCI is to make researchers focus on improving their own BCI algorithms and performing their experiments without wasting too much time on preliminary preparations.

## Features

* Improvements to MOABB APIs
   - add hook functions to control the preprocessing flow more easily
   - use joblib to accelerate the data loading
   - add proxy options for network connection issues
   - add more information in the meta of data
   - other small changes

* Supported Datasets
   - MI Datasets
     - AlexMI
     - BNCI2014001, BNCI2014004
     - PhysionetMI, PhysionetME
     - Cho2017
     - MunichMI
     - Schirrmeister2017
     - Weibo2014
     - Zhou2016
   - SSVEP Datasets
     - Nakanishi2015
     - Wang2016
     - BETA

* Implemented BCI algorithms
   - Decomposition Methods
     - SPoC, CSP, MultiCSP and FBCSP
     - CCA, itCCA, MsCCA, ExtendCCA, ttCCA, MsetCCA, MsetCCA-R, TRCA, TRCA-R, SSCOR and TDCA
     - DSP
   - Manifold Learning
     - Basic Riemannian Geometry operations
     - Alignment methods
     - Riemann Procustes Analysis
   - Deep Learning
     - ShallowConvNet
     - EEGNet
     - ConvCA
     - GuneyNet
     - Cross dataset transfer learning based on pre-training
   - Transfer Learning
     - MEKT
     - LST

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TBC-TJU/MetaBCI.git
   ```
2. Change to the project directory
   ```sh
   cd MetaBCI
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```
## Who are we?

The MetaBCI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China
- Tianjin Brain Center, China


## What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation and technical writing and project management.

We'd love your feedback along the way.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: TBC_TJU_2022@163.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
