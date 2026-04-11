import argparse
from model import build_network, train_network
from utils import setup_device, create_directories, save_accuracy_to_csv, save_weight
import torch
import os


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_neurons", type=int, default=400)
    parser.add_argument("--input_neuron", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--theta_plus", type=float, default=0.05)
    parser.add_argument("--time", type=int, default=250)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--intensity", type=float, default=64)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--update_interval", type=int, default=250)
    parser.add_argument("--update_inhibation_weights", type=int, default=500)
    parser.add_argument("--plot_interval", type=int, default=250)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.add_argument("--cuda_name", type=str, default="cuda:0")
    parser.add_argument("--lsm_out_name", type=str, default="CIR_50_3w_train_100_3.pt")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.set_defaults(plot=True, gpu=True)
    args = parser.parse_args()

    # Set up device
    device = setup_device(args.cuda_name, args.gpu, args.seed)

    # Build network
    network, spike_record, spikes, voltages, accuracy, assignments, proportions, rates, plt_sqrt, som_voltage_monitor = build_network(
        input_neuron=args.input_neuron,
        n_neurons=args.n_neurons,
        theta_plus=args.theta_plus,
        time=args.time,
        dt=args.dt,
        device=device,
        update_interval=args.update_interval,
        n_classes=args.n_classes
    )

    # diagonal weights for increasing the inhibition
    weights_mask = (1 - torch.diag(torch.ones(args.n_neurons))).to(device)

    # Determine task name
    task_type = 'Ga_Multi'
    plot_name = f"{os.path.basename(args.lsm_out_name).replace('.pt', '')}_{args.n_neurons}_epoch_{args.n_epochs}"
    create_directories(task_type, plot_name)

    # Training and evaluation
    accuracy, assignments, proportions = train_network(
        network=network,
        spike_record=spike_record,
        spikes=spikes,
        som_voltage_monitor=som_voltage_monitor,
        accuracy=accuracy,
        assignments=assignments,
        proportions=proportions,
        rates=rates,
        args=args,
        device=device,
        weights_mask=weights_mask,
        plt_sqrt=plt_sqrt,
        plot_name=plot_name,
        task_type=task_type,
        n_classes=args.n_classes,
    )

    # Save accuracy and model data
    save_accuracy_to_csv(plot_name, accuracy)
    save_weight(plot_name, assignments, proportions)


if __name__ == "__main__":
    main()
