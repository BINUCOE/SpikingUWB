# Spiking UWB 

Code for "Exploring the Potential of Spiking Neural Networks in UWB ULOS Identification"


## Data
The raw data used in this repo is available at the following link: https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set

The processed data is in the `data` folder, with the following tree structure:

```data
data
├── cir_50
│   ├── 1.text
│   ├── 2.text
│   ├── 3.text
│   ├── 4.text
│   ├── 5.text
│   ├── 6.text
│   └── 7.text
└── cir_120
    ├── 1.text
    ├── 2.text
    ├── 3.text
    ├── 4.text
    ├── 5.text
    ├── 6.text
    └── 7.text
```

## Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
Liquid encoder, use the following command:

```bash
python liquid_encoder.py --mode=seq_encode --length=<length_of_sequence>
```

To train the model, use the following command:

```bash
python train.py --cuda_name=cuda:0 --lsm_out_name=<output path of liquid> --input_neuron=<input neuron number> --n_neurons=<readout neuron> --time=250 --n_epochs=<number of epochs>
```

## Citation
If you find this code useful for your research, please consider citing the following paper:

```
@article{zhang2025exploring,
  title={Exploring the Potential of Spiking Neural Networks in UWB ULOS Identification},
  author={Youdong Zhang, Xu He and Xiaolin Meng},
  journal={arXiv preprint arXiv:2512.23975},
  year={2025}
}
```

---
<a href="https://clustrmaps.com/site/1c9re"  title="ClustrMaps">
    <img src="//www.clustrmaps.com/map_v2.png?d=yQ4sUjzNVbmf6220VOIXauNfWZ-57o6oexp4_u4HzRQ&cl=ffffff" />
</a>