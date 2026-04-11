import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Gamma
from torch.utils.data import DataLoader
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import UWB
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")

parser.add_argument("--cuda_name", type=str, default="cuda:0")
parser.add_argument("--mode", type=str, default="seq_encode")
parser.add_argument("--length", type=int, default=50)
parser.add_argument("--mlsm", type=int, nargs='+', default=[500, 400])
parser.set_defaults(plot=False, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_workers = args.n_workers
dt = args.dt
intensity = args.intensity
train = args.train
mode = args.mode
plot = args.plot
gpu = args.gpu
cuda_name = args.cuda_name
MLSM = args.mlsm
length = args.length

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

path_dic = {
    "raw_50": "data/cir_50",
    "raw_120": "data/cir_120",
}

if mode in ['seq_encode', 'single_cir_seq_encode', 'single_rf_seq_encode']:
    pass
else:
    raise ValueError(f"Unknown mode type: {mode}")

if train:
    save_name = f'CIR_{length}_train_3w_gamma_{mode}_real_raw_{sum(MLSM)}_{len(MLSM)}'
else:
    save_name = f'Edge_CIR_{length}_test_3w_gamma_{mode}_real_raw_{sum(MLSM)}_{len(MLSM)}'

print(save_name)

# Sets up Gpu use
device = torch.device(f"{cuda_name}" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0
else:
    n_workers = torch.cuda.is_available() * 8 * torch.cuda.device_count()

# Create simple Torch NN
network = Network(dt=dt)

def set_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (3.5, 2.5),  # IEEE单栏宽度
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.8,
        'lines.markersize': 3,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })

set_ieee_style()

class DualUWB:
    """
    双路UWB数据集类
    """

    def __init__(self, uwb_cir, uwb_rf, label, uwb_encoder):
        self.uwb_cir = uwb_cir
        self.uwb_rf = uwb_rf
        self.label = label
        self.uwb_encoder = uwb_encoder

    def __len__(self):
        return len(self.uwb_cir)

    def __getitem__(self, idx):
        encoded_uwb_cir = self.uwb_encoder(self.uwb_cir[idx])
        encoded_uwb_rf = self.uwb_encoder(self.uwb_rf[idx])


        return {
            "encoded_cir": encoded_uwb_cir,
            "encoded_rf": encoded_uwb_rf,
            "label": self.label[idx],
        }


def single_lsm_gamma(MLSM, input_name, connect_name, x, y):
    alpha = 2
    beta = 1

    inpt = Input(x * y, shape=(1, x, y))
    network.add_layer(inpt, name=input_name)

    output_1 = LIFNodes(MLSM, thresh=-52 + np.random.randn(MLSM).astype(float))
    network.add_layer(output_1, name=connect_name)

    weight_C1_1 = 0.5 * Gamma(concentration=alpha, rate=beta).sample((inpt.n, output_1.n))
    negative_mask = torch.rand(inpt.n, output_1.n) < 0.5
    weight_C1_1[negative_mask] = -weight_C1_1[negative_mask]
    weight_C2_1 = 0.5 * Gamma(concentration=alpha, rate=beta).sample((output_1.n, output_1.n))
    negative_mask = torch.randn(output_1.n, output_1.n) < 0.5
    weight_C2_1[negative_mask] = -weight_C2_1[negative_mask]

    C1_1 = Connection(source=inpt, target=output_1, w=weight_C1_1)
    C2_1 = Connection(source=output_1, target=output_1, w=weight_C2_1)

    network.add_connection(C1_1, source=input_name, target=connect_name)
    network.add_connection(C2_1, source=connect_name, target=connect_name)


def single_lsm_randn(MLSM, input_name, connect_name, x, y):
    inpt = Input(x * y, shape=(1, x, y))
    network.add_layer(inpt, name=input_name)

    output = LIFNodes(MLSM, thresh=-52 + np.random.randn(MLSM).astype(float))
    network.add_layer(output, name=connect_name)
    C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
    C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))

    network.add_connection(C1, source=input_name, target=connect_name)
    network.add_connection(C2, source=connect_name, target=connect_name)


from UWB_Processor import UWBDataProcessor

processor = UWBDataProcessor(path_dic[f"raw_{length}"], length=length, train=train)

cir_tensor, rf_tensor, label_tensor = processor.process_time_domain()


dataset = DualUWB(uwb_cir=cir_tensor,
                  uwb_rf=rf_tensor,
                  label=label_tensor,
                  uwb_encoder=PoissonEncoder(time=args.time, dt=dt))

dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=n_workers
     )


if mode == "seq_encode":
    # seq encode real raw
    single_lsm_gamma(MLSM[0], input_name="I_1", connect_name="O_1", x=1, y=length)
    single_lsm_gamma(MLSM[1], input_name="I_2", connect_name="O_2", x=1, y=10)

elif mode == "single_cir_seq_encode":
    # single cir seq encode
    single_lsm_gamma(MLSM[0], input_name="I_1", connect_name="O_1", x=1, y=length)

elif mode == "single_rf_seq_encode":
    # single rf seq encode
    single_lsm_gamma(MLSM[0], input_name="I_1", connect_name="O_1", x=1, y=10)

else:
    raise ValueError(f"Unknown mode type: {mode}")


spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=args.time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O_1": Monitor(network.layers["O_1"], ["v"], time=args.time, device=device)}
network.add_monitor(voltages["O_1"], name="O_1_voltages")

network.to(device)

spike_axes = None
spike_ims = None

n_iters = len(dataloader)
training_pairs = []
pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break

    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

    if mode == "seq_encode":
        # seq encode real raw
        datum_cir = dataPoint["encoded_cir"].view(int(args.time / dt), 1, 1, length).to(device)
        datum_rf = dataPoint["encoded_rf"].view(int(args.time / dt), 1, 1, 10).to(device)

        network.run(inputs={"I_1": datum_cir, "I_2": datum_rf}, time=args.time)
        cir_spikes = spikes["O_1"].get("s")
        rf_spikes = spikes["O_2"].get("s")
        concatenated_spikes = torch.concat((cir_spikes, rf_spikes), dim=2)
        training_pairs.append([concatenated_spikes, label])

    elif mode == "single_cir_seq_encode":
        # single cir seq encode
        datum_cir = dataPoint["encoded_cir"].view(int(args.time / dt), 1, 1, length).to(device)

        network.run(inputs={"I_1": datum_cir}, time=args.time)
        out_spike = spikes["O_1"].get("s")
        training_pairs.append([out_spike, label])

    elif mode == "single_rf_seq_encode":
        # single rf seq encode
        datum_rf = dataPoint["encoded_rf"].view(int(args.time / dt), 1, 10, 10).to(device)

        network.run(inputs={"I_1": datum_rf}, time=args.time)
        out_spike = spikes["O_1"].get("s")
        training_pairs.append([out_spike, label])

    else:
        raise ValueError(f"Unknown mode type: {mode}")


    if plot:
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(args.time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )

        plt.pause(1e-8)
    network.reset_state_variables()

training_pairs_dicts = [
    {"encoded_cir": spikes, "label": label}
    for spikes, label in training_pairs
]

if not os.path.exists("Liquid_Out"):
    os.makedirs("Liquid_Out")

torch.save(training_pairs_dicts, open('Liquid_Out/{}.pt'.format(save_name), "wb"))
