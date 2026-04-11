import torch
import numpy as np
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from tqdm import tqdm
from time import time as t
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.utils import get_square_assignments, get_square_weights
import matplotlib.pyplot as plt


def build_network(input_neuron, n_neurons, theta_plus, time, dt, device, update_interval, n_classes):
    plt_sqrt = int(np.ceil(np.sqrt(input_neuron)))
    network = IncreasingInhibitionNetwork(
        n_input=input_neuron,
        n_neurons=n_neurons,
        start_inhib=10,
        max_inhib=-40.0,
        theta_plus=theta_plus,
        tc_theta_decay=1e7,
        inpt_shape=(1, plt_sqrt, plt_sqrt),
        nu=(1e-4, 1e-2),
    )
    network.to(device)

    spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)
    accuracy = {"all": [], "proportion": []}

    som_voltage_monitor = Monitor(
        network.layers["Y"], ["v"], time=int(time / dt), device=device
    )
    network.add_monitor(som_voltage_monitor, name="som_voltage")

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name=f"{layer}_spikes")

    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
        )
        network.add_monitor(voltages[layer], name=f"{layer}_voltages")

    return network, spike_record, spikes, voltages, accuracy, assignments, proportions, rates, plt_sqrt, som_voltage_monitor


def train_network(network, spike_record, spikes, som_voltage_monitor, plt_sqrt, accuracy, assignments, proportions, rates, args,
                  device, weights_mask, plot_name, task_type, n_classes):

    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None

    n_sqrt = int(np.ceil(np.sqrt(args.n_neurons)))
    save_weights_fn = f"{task_type}/{plot_name}/weights/weights.png"
    save_performance_fn = f"{task_type}/{plot_name}/performance/performance.png"
    save_assaiments_fn = f"{task_type}/{plot_name}/assaiments/assaiments.png"

    print("\nBegin training.\n")
    start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.load(open(f"{args.lsm_out_name}", "rb"), map_location=device)

    dataloader = torch.utils.data.DataLoader(
        dataloader, batch_size=1, shuffle=True
    )

    for epoch in range(args.n_epochs):
        labels = []

        if epoch % args.progress_interval == 0:
            print(f"Progress: {epoch} / {args.n_epochs} ({t() - start:.4f} seconds)")
            start = t()

        pbar = tqdm(total=len(dataloader))
        for step, batch in enumerate(dataloader):
            if step == len(dataloader):
                break

            inputs = {
                "X": batch['encoded_cir'].view(int(args.time / args.dt), 1, 1, plt_sqrt, plt_sqrt).to(device)
            }

            if step > 0:
                if step % args.update_inhibation_weights == 0:
                    if step % (args.update_inhibation_weights * 10) == 0:
                        network.Y_to_Y.w -= weights_mask * 50
                    else:
                        network.Y_to_Y.w -= weights_mask * 0.5

                if step % args.update_interval == 0:
                    label_tensor = torch.tensor(labels, device=device)
                    all_activity_pred = all_activity(spike_record, assignments, n_labels=n_classes)
                    proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_labels=n_classes)

                    accuracy["all"].append(
                        100 * torch.sum(torch.tensor(label_tensor.long() == all_activity_pred)).item() / len(label_tensor)
                    )
                    accuracy["proportion"].append(
                        100 * torch.sum(torch.tensor(label_tensor.long() == proportion_pred)).item() / len(label_tensor)
                    )

                    tqdm.write(
                        f"\nAll activity accuracy: {accuracy['all'][-1]:.2f} (last), "
                        f"{np.mean(accuracy['all']):.2f} (average), {np.max(accuracy['all']):.2f} (best)"
                    )
                    tqdm.write(
                        f"Proportion weighting accuracy: {accuracy['proportion'][-1]:.2f} (last), "
                        f"{np.mean(accuracy['proportion']):.2f} (average), {np.max(accuracy['proportion']):.2f} (best)\n"
                    )

                    assignments, proportions, rates = assign_labels(
                        spikes=spike_record,
                        labels=label_tensor,
                        n_labels=n_classes,
                        rates=rates
                    )

                    if accuracy["all"][-1] == np.max(accuracy["all"]):
                        print("Saving model")
                        network.save(f'model_weights/model/network_{plot_name}.pt')

                    labels = []

            labels.append(batch['label'])

            temp_spikes = 0
            for retry in range(5):
                network.run(inputs=inputs, time=args.time)
                temp_spikes = spikes["Y"].get("s").squeeze()

                if temp_spikes.sum().sum() < 2:
                    inputs["X"] *= batch['encoded_cir'].view(
                        int(args.time / args.dt), 1, 1, plt_sqrt, plt_sqrt
                    ).to(device)
                else:
                    break

            exc_voltages = som_voltage_monitor.get("v")
            spike_record[step % args.update_interval].copy_(temp_spikes, non_blocking=True)

            if args.plot and step % args.plot_interval == 0:
                # audio = batch['audio'].view(28, 28)
                inpt = inputs["X"].view(args.time, args.input_neuron).sum(0).view(plt_sqrt, plt_sqrt)
                input_exc_weights = network.connections[("X", "Y")].w
                square_weights = get_square_weights(
                    input_exc_weights.view(args.input_neuron, args.n_neurons), n_sqrt, plt_sqrt
                )
                square_assignments = get_square_assignments(assignments, n_sqrt)
                spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
                voltages = {"Y": exc_voltages}
                # inpt_axes, inpt_ims = plot_input(
                #     audio, inpt, label=batch['label'], axes=inpt_axes, ims=inpt_ims
                # )
                spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
                # [weights_im, save_weights_fn] = plot_weights(
                #     square_weights, im=weights_im, save=save_weights_fn
                # )
                assigns_im = plot_assignments(
                    square_assignments, im=assigns_im, save=save_assaiments_fn
                )
                perf_ax = plot_performance(accuracy, ax=perf_ax, save=save_performance_fn)
                voltage_ims, voltage_axes = plot_voltages(
                    voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
                )
                #
                plt.pause(1e-8)

            network.reset_state_variables()
            pbar.set_description_str("Train progress: ")
            pbar.update()

        print(f"\n Progress: {epoch + 1} / {args.n_epochs} ({t() - start:.4f} seconds)")

    return accuracy, assignments, proportions
