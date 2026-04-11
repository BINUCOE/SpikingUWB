import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

from bindsnet.evaluation import all_activity, proportion_weighting
from bindsnet.network.monitors import Monitor
from bindsnet.network import load


# ---------------------------------------------------------------------------
# 能耗参数常量（参考 Intel Loihi / 典型 CMOS 工艺估算值）
# ---------------------------------------------------------------------------
E_SOP_PJ      = 23.6    # 每次突触操作能耗 (pJ)，Loihi 参考值
E_LIF_STEP_PJ = 0.9     # 每个神经元每时间步 LIF 更新能耗 (pJ)
W_BITS        = 8       # 权重位宽 (bits)
E_SRAM_BIT_PJ = 50.0    # SRAM 读取能耗 (pJ/bit)


def set_ieee_style():
    """设置IEEE论文风格的绘图参数"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': (3.5, 2.5),
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
    """双路UWB数据集类"""

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


class BinaryClassificationTester:
    """
    二分类测试器：在原有评估基础上新增
      - 逐样本推理时间统计
      - 基于脉冲稀疏性的三分量能耗估算
          E_syn  = spikes × E_SOP
          E_neu  = N_neurons × T_steps × E_LIF_STEP
          E_mem  = spikes × W_bits × E_SRAM_bit
    """

    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.network = None
        self.spikes = {}
        self.voltages = {}

        # 分类精度
        self.accuracy = {"all": 0, "proportion": 0}
        self.actual_labels = []
        self.predict_labels = []
        self.probability_scores = []

        # ---- 新增：推理时间与能耗记录 ----
        self.inference_times   = []   # 每条样本的推理时间 (s)
        self.spike_counts      = []   # 每条样本的总脉冲数
        self.energy_syn_pj     = []   # 突触操作能耗 (pJ)
        self.energy_neu_pj     = []   # 神经元更新能耗 (pJ)
        self.energy_mem_pj     = []   # 内存访问能耗 (pJ)
        self.energy_total_pj   = []   # 单样本总能耗 (pJ)
        # -----------------------------------


    def _setup_device(self):
        device = torch.device(f"{self.args.cuda_name}" if torch.cuda.is_available() else "cpu")
        if self.args.gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            torch.manual_seed(self.args.seed)
            device = "cpu"
            if self.args.gpu:
                self.args.gpu = False
        torch.set_num_threads(os.cpu_count() - 1)
        print(f"Running on Device = {device}")
        return device

    def load_network(self):
        network_name = f'model_weights/model/network_{self.args.plot_name}.pt'
        self.network = load(network_name, learning=False, map_location=self.device)
        self.network.to(self.device)
        print("\n=========Loading network complete.=========")
        self._setup_monitors()

    def _setup_monitors(self):
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer],
                state_vars=["s"],
                time=int(self.args.time / self.args.dt),
                device=self.device
            )
            self.network.add_monitor(self.spikes[layer], name=f"{layer}_spikes")

        for layer in set(self.network.layers) - {"X"}:
            self.voltages[layer] = Monitor(
                self.network.layers[layer],
                state_vars=["v"],
                time=int(self.args.time / self.args.dt),
                device=self.device
            )
            self.network.add_monitor(self.voltages[layer], name=f"{layer}_voltages")

    def load_data(self):
        # from UWB_Processor import UWBDataProcessor
        # from bindsnet.encoding import PoissonEncoder
        # processor = UWBDataProcessor("data/cir_50", length=50, train=False)
        # cir_tensor, rf_tensor, label_tensor = processor.process_time_domain()
        # dataset = DualUWB(
        #     uwb_cir=cir_tensor,
        #     uwb_rf=rf_tensor,
        #     label=label_tensor,
        #     uwb_encoder=PoissonEncoder(time=250, dt=1.0)
        # )
        # dataloader = torch.utils.data.DataLoader(
        #     dataset, batch_size=1, shuffle=True, num_workers=4
        # )
        dataloader = torch.load(open(f"{self.args.lsm_out_data}", "rb"), map_location=self.device)
        import random
        random.shuffle(dataloader)
        return dataloader

    def load_assignments(self):
        assignments_name = f'model_weights/weight/assignments_{self.args.plot_name}_last.npy'
        proportions_name = f'model_weights/weight/proportions_{self.args.plot_name}_last.npy'
        assignments = torch.nn.Parameter(
            torch.from_numpy(np.load(assignments_name)).float().to(self.device),
            requires_grad=False
        )
        proportions = torch.nn.Parameter(
            torch.from_numpy(np.load(proportions_name)).float().to(self.device),
            requires_grad=False
        )
        return assignments, proportions

    # ------------------------------------------------------------------
    # 核心推理循环（含时间与能耗采集）
    # ------------------------------------------------------------------
    def run_testing(self, dataloader, assignments, proportions):
        print("\n=========Begin testing.=========")
        self.network.train(mode=False)

        T_steps   = int(self.args.time / self.args.dt)   # 仿真时间步数
        N_neurons = self.args.n_neurons                   # 输出层神经元数

        spike_record = torch.zeros(1, T_steps, N_neurons, device=self.device)
        test_Y_index = []
        global_start = t()

        pbar = tqdm(total=len(dataloader))

        for step, batch in enumerate(dataloader):
            if step >= len(dataloader):
                break

            inputs = self._prepare_inputs(batch)

            # ---- 推理计时开始 ----
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            sample_start = t()

            self.network.run(inputs=inputs, time=self.args.time, device=self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            sample_end = t()
            # ---- 推理计时结束 ----
            # 单个样本的推理时间
            inference_time = sample_end - sample_start
            self.inference_times.append(inference_time)

            # 获取输出层脉冲张量
            spike_record[0] = self.spikes["Y"].get("s").squeeze()

            # ---- 能耗估算 ----
            n_spikes = int(spike_record[0].sum().item())   # 总放电次数
            self.spike_counts.append(n_spikes)

            e_syn  = n_spikes * E_SOP_PJ                           # 突触操作
            e_neu  = N_neurons * T_steps * E_LIF_STEP_PJ           # LIF 更新（全神经元）
            e_mem  = n_spikes * W_BITS * E_SRAM_BIT_PJ             # 权重内存访问
            e_tot  = e_syn + e_neu + e_mem

            self.energy_syn_pj.append(e_syn)
            self.energy_neu_pj.append(e_neu)
            self.energy_mem_pj.append(e_mem)
            self.energy_total_pj.append(e_tot)
            # ----------------------

            label_tensor = torch.tensor(batch['label'], device=self.device).squeeze()

            if self.args.tsne:
                activation_counts = torch.sum(spike_record[0], dim=0)
                most_active_neuron_index = torch.argmax(activation_counts)
                test_Y_index.append([label_tensor, most_active_neuron_index])

            all_activity_pred, proportion_pred = self._get_predictions(
                spike_record, assignments, proportions, self.args.n_classes
            )

            self._update_accuracy(label_tensor, all_activity_pred, proportion_pred)
            self.actual_labels.append(label_tensor.cpu().item())
            self.predict_labels.append(all_activity_pred.cpu().item())

            probability = self._compute_probability_scores(spike_record, assignments)
            self.probability_scores.append(probability)


            self.network.reset_state_variables()
            pbar.set_description_str(f"Test progress: {step + 1}/{len(dataloader)}")
            pbar.update()

        pbar.close()
        total_time = t() - global_start
        print(f"\nTesting completed in {total_time:.2f} seconds "
              f"({len(dataloader)} samples, "
              f"avg {np.mean(self.inference_times)*1000:.2f} ms/sample)")

        if self.args.tsne:
            torch.save(test_Y_index, open(f'test_Y_index_{self.args.plot_name}.pt', "wb"))

    def _prepare_inputs(self, batch):
        plt_sqrt = int(np.ceil(np.sqrt(self.args.input_neuron)))
        if self.args.task == 'audio':
            inputs = {"X": batch["encoded_audio"].view(int(self.args.time / self.args.dt), 1, 1, plt_sqrt, plt_sqrt)}
        elif self.args.task == 'image':
            inputs = {"X": batch["encoded_image"].view(int(self.args.time / self.args.dt), 1, 1, plt_sqrt, plt_sqrt)}
        elif self.args.task == 'uwb':
            inputs = {"X": batch["encoded_cir"].view(int(self.args.time / self.args.dt), 1, 1, plt_sqrt, plt_sqrt)}
        else:
            raise ValueError("Invalid task type. Expected 'audio', 'image', or 'uwb'.")

        if self.args.gpu:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _get_predictions(self, spike_record, assignments, proportions, n_classes):
        all_activity_pred = all_activity(
            spikes=spike_record, assignments=assignments, n_labels=n_classes
        ).squeeze()
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )
        return all_activity_pred, proportion_pred

    def _update_accuracy(self, label, all_activity_pred, proportion_pred):
        self.accuracy["all"] += float(
            torch.sum(torch.tensor(label.long() == all_activity_pred.to(self.device))).item()
        )
        self.accuracy["proportion"] += float(
            torch.sum(torch.tensor(label.long() == proportion_pred.to(self.device))).item()
        )

    def _compute_probability_scores(self, spike_record, assignments):
        spike_counts = torch.sum(spike_record[0], dim=0)
        class_spikes = []
        for i in range(self.args.n_classes):
            mask = (assignments == i)
            class_spikes.append(torch.sum(spike_counts * mask.float()))
        class_spikes = torch.tensor(class_spikes)
        probabilities = torch.softmax(class_spikes, dim=0)
        return probabilities.cpu().numpy()

    # ------------------------------------------------------------------
    # 指标汇总（含推理时间与能耗报告）
    # ------------------------------------------------------------------
    def compute_metrics(self, n_test):
        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)

        all_activity_acc = self.accuracy["all"] / n_test
        proportion_acc   = self.accuracy["proportion"] / n_test

        print(f"All activity accuracy:         {all_activity_acc:.4f}")
        print(f"Proportion weighting accuracy: {proportion_acc:.4f}")

        y_true = np.array(self.actual_labels)
        y_pred = np.array(self.predict_labels)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall[1]

        print("\nPer-class Metrics:")
        print(f"  Class 0 - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
        print(f"  Class 1 - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")

        print("\nAggregate Metrics:")
        print(f"  Macro    - P: {precision_macro:.4f}, R: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"  Weighted - P: {precision_weighted:.4f}, R: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")

        print(f"\nBinary Classification:")
        print(f"  Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}")
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))

        # ---- 推理时间统计 ----
        inf_arr = np.array(self.inference_times) * 1000   # 转为毫秒
        print("=" * 50)
        print("INFERENCE TIME (ms/sample)")
        print("=" * 50)
        print(f"  Mean   : {inf_arr.mean():.3f} ms")
        print(f"  Std    : {inf_arr.std():.3f} ms")
        print(f"  Min    : {inf_arr.min():.3f} ms")
        print(f"  Max    : {inf_arr.max():.3f} ms")
        print(f"  Median : {np.median(inf_arr):.3f} ms")
        print(f"  Total  : {inf_arr.sum()/1000:.3f} s")

        # ---- 能耗统计 ----
        sc_arr   = np.array(self.spike_counts)
        T_steps  = int(self.args.time / self.args.dt)
        N        = self.args.n_neurons
        total_possible = N * T_steps   # 最大可能脉冲数

        e_syn_arr = np.array(self.energy_syn_pj)
        e_neu_arr = np.array(self.energy_neu_pj)
        e_mem_arr = np.array(self.energy_mem_pj)
        e_tot_arr = np.array(self.energy_total_pj)

        print("\n" + "=" * 50)
        print("POWER / ENERGY ESTIMATION")
        print("=" * 50)
        print(f"  Energy model parameters:")
        print(f"    E_SOP      = {E_SOP_PJ} pJ/spike")
        print(f"    E_LIF_step = {E_LIF_STEP_PJ} pJ/neuron/step")
        print(f"    Weight bits= {W_BITS} bit  E_SRAM = {E_SRAM_BIT_PJ} pJ/bit")
        print(f"    N_neurons  = {N},  T_steps = {T_steps}")

        print(f"\n  Spike statistics (output layer Y):")
        print(f"    Mean spikes/sample : {sc_arr.mean():.1f}  "
              f"(firing rate = {sc_arr.mean()/total_possible*100:.2f}%)")
        print(f"    Std                : {sc_arr.std():.1f}")
        print(f"    Min / Max          : {sc_arr.min()} / {sc_arr.max()}")

        print(f"\n  Energy per sample (pJ):")
        print(f"    E_syn  (synaptic ops)  : {e_syn_arr.mean():.1f} ± {e_syn_arr.std():.1f}")
        print(f"    E_neu  (LIF updates)   : {e_neu_arr.mean():.1f} ± {e_neu_arr.std():.1f}  [fixed]")
        print(f"    E_mem  (weight SRAM)   : {e_mem_arr.mean():.1f} ± {e_mem_arr.std():.1f}")
        print(f"    E_total                : {e_tot_arr.mean():.1f} ± {e_tot_arr.std():.1f}")
        print(f"    E_total (nJ)           : {e_tot_arr.mean()/1000:.3f} nJ")

        # 瞬时功率估算：P = E_total / T_sim
        T_sim_s = self.args.time * 1e-3   # time 单位为 ms，转为秒
        p_mean_uw = e_tot_arr.mean() * 1e-12 / T_sim_s * 1e6   # 转为 µW
        print(f"\n  Estimated average power:")
        print(f"    T_sim = {self.args.time} ms")
        print(f"    P_avg ≈ {p_mean_uw:.4f} µW  "
              f"({p_mean_uw*1000:.4f} nW)")

        # ================================================================
        # Bootstrap 95% 置信区间（非参数，B=2000 次重采样）
        # 对 Accuracy、Macro F1、Sensitivity、Specificity 同时估计
        # ================================================================
        B   = 2000
        rng = np.random.default_rng(self.args.seed)
        n   = len(y_true)

        boot_acc, boot_f1, boot_sens, boot_spec = [], [], [], []
        for _ in range(B):
            idx  = rng.integers(0, n, size=n)          # 有放回重采样
            yt_b = y_true[idx]
            yp_b = y_pred[idx]
            boot_acc.append(np.mean(yt_b == yp_b))
            _, _, f1_b, _ = precision_recall_fscore_support(
                yt_b, yp_b, average='macro', zero_division=0)
            boot_f1.append(f1_b)
            cm_b = confusion_matrix(yt_b, yp_b, labels=[0, 1])
            tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
            boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else np.nan)
            boot_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else np.nan)

        boot_acc  = np.array(boot_acc)
        boot_f1   = np.array(boot_f1)
        boot_sens = np.array(boot_sens)
        boot_spec = np.array(boot_spec)

        def ci95(arr):
            arr = arr[~np.isnan(arr)]
            return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

        acc_lo,  acc_hi  = ci95(boot_acc)
        f1_lo,   f1_hi   = ci95(boot_f1)
        sens_lo, sens_hi = ci95(boot_sens)
        spec_lo, spec_hi = ci95(boot_spec)

        print("\n" + "=" * 50)
        print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
        print("=" * 50)
        print(f"  B = {B} resamplings,  n = {n} samples")
        print(f"  Accuracy    : {all_activity_acc:.4f}  [{acc_lo:.4f}, {acc_hi:.4f}]")
        print(f"  Macro F1    : {f1_macro:.4f}  [{f1_lo:.4f}, {f1_hi:.4f}]")
        print(f"  Sensitivity : {sensitivity:.4f}  [{sens_lo:.4f}, {sens_hi:.4f}]")
        print(f"  Specificity : {specificity:.4f}  [{spec_lo:.4f}, {spec_hi:.4f}]")

        metrics = {
            'accuracy_all':         all_activity_acc,
            'accuracy_proportion':  proportion_acc,
            'precision_per_class':  precision,
            'recall_per_class':     recall,
            'f1_per_class':         f1,
            'precision_macro':      precision_macro,
            'recall_macro':         recall_macro,
            'f1_macro':             f1_macro,
            'precision_weighted':   precision_weighted,
            'recall_weighted':      recall_weighted,
            'f1_weighted':          f1_weighted,
            'sensitivity':          sensitivity,
            'specificity':          specificity,
            'confusion_matrix':     [[tn, fp], [fn, tp]],
            # 能耗与时间
            'inference_time_ms':    inf_arr,
            'spike_counts':         sc_arr,
            'energy_syn_pj':        e_syn_arr,
            'energy_neu_pj':        e_neu_arr,
            'energy_mem_pj':        e_mem_arr,
            'energy_total_pj':      e_tot_arr,
            'power_avg_uw':         p_mean_uw,
            # Bootstrap 置信区间
            'bootstrap_acc_ci':     (acc_lo,  acc_hi),
            'bootstrap_f1_ci':      (f1_lo,   f1_hi),
            'bootstrap_sens_ci':    (sens_lo, sens_hi),
            'bootstrap_spec_ci':    (spec_lo, spec_hi),
            'bootstrap_samples':    {'acc': boot_acc, 'f1': boot_f1,
                                     'sens': boot_sens, 'spec': boot_spec},
        }
        return metrics

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def plot_ieee_confusion_matrix(self, metrics, save_path=None):
        cm = np.array(metrics['confusion_matrix'])
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
        blues_cmap = plt.cm.Blues
        im = ax.imshow(cm, cmap=blues_cmap, interpolation='nearest', vmin=0, vmax=cm.max())
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['LOS', 'NLOS'], fontsize=9)
        ax.set_yticklabels(['LOS', 'NLOS'], fontsize=9)
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{int(cm[i, j])}",
                        ha="center", va="center",
                        color=color, fontsize=10, fontweight='bold')
        plt.tight_layout()
        if save_path and self.args.save_confusion_matrix:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.show()

    def plot_ieee_metrics_comparison(self, metrics, save_path=None):
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        class_0_metrics = [metrics['accuracy_all'], metrics['precision_per_class'][0],
                           metrics['recall_per_class'][0], metrics['f1_per_class'][0]]
        class_1_metrics = [metrics['accuracy_all'], metrics['precision_per_class'][1],
                           metrics['recall_per_class'][1], metrics['f1_per_class'][1]]
        macro_metrics   = [metrics['accuracy_all'], metrics['precision_macro'],
                           metrics['recall_macro'],  metrics['f1_macro']]
        x     = np.arange(len(categories))
        width = 0.25
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        rects1 = ax.bar(x - width, class_0_metrics, width, label='Class 0', edgecolor='black')
        rects2 = ax.bar(x,         class_1_metrics, width, label='Class 1', edgecolor='black')
        rects3 = ax.bar(x + width, macro_metrics,   width, label='Macro Avg', edgecolor='black')
        ax.set_xlabel('Metrics', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Classification Performance Metrics', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8, frameon=True, fancybox=False, framealpha=0.7)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)
        autolabel(rects1); autolabel(rects2); autolabel(rects3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Metrics comparison saved to: {save_path}")
        plt.show()

    def plot_ieee_power_analysis(self, metrics, save_path=None):
        """
        IEEE 风格的能耗分析图（2×2 布局）：
          左上：推理时间分布直方图
          右上：能耗三分量叠加柱状图（按样本索引抽样）
          左下：每样本总能耗折线图
          右下：脉冲数 vs 总能耗散点图
        """
        inf_ms  = metrics['inference_time_ms']
        e_syn   = metrics['energy_syn_pj']
        e_neu   = metrics['energy_neu_pj']
        e_mem   = metrics['energy_mem_pj']
        e_tot   = metrics['energy_total_pj']
        spikes  = metrics['spike_counts']

        fig, axes = plt.subplots(2, 2, figsize=(7, 5))
        fig.suptitle('Inference Time & Energy Analysis', fontsize=11)

        # --- 左上：推理时间直方图 ---
        ax = axes[0, 0]
        ax.hist(inf_ms, bins=30, edgecolor='black', linewidth=0.4)
        ax.axvline(inf_ms.mean(), color='red', linestyle='--', linewidth=0.8,
                   label=f'Mean={inf_ms.mean():.2f}ms')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Latency Distribution')
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        # --- 右上：能耗三分量叠加柱状图（最多显示 60 条） ---
        ax = axes[0, 1]
        n_show  = min(60, len(e_tot))
        step_   = max(1, len(e_tot) // n_show)
        idx     = np.arange(0, len(e_tot), step_)[:n_show]
        ax.bar(range(len(idx)), e_syn[idx] / 1e3,  label='E_syn',  edgecolor='none')
        ax.bar(range(len(idx)), e_neu[idx] / 1e3,  label='E_neu',
               bottom=e_syn[idx] / 1e3, edgecolor='none')
        ax.bar(range(len(idx)), e_mem[idx] / 1e3,  label='E_mem',
               bottom=(e_syn[idx] + e_neu[idx]) / 1e3, edgecolor='none')
        ax.set_xlabel('Sample Index (sampled)')
        ax.set_ylabel('Energy (nJ)')
        ax.set_title('Energy Breakdown per Sample')
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(True, linewidth=0.3, alpha=0.5, axis='y')

        # --- 左下：每样本总能耗折线 ---
        ax = axes[1, 0]
        ax.plot(e_tot / 1e3, linewidth=0.6, color='steelblue')
        ax.axhline(e_tot.mean() / 1e3, color='red', linestyle='--', linewidth=0.8,
                   label=f'Mean={e_tot.mean()/1e3:.3f} nJ')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Total Energy (nJ)')
        ax.set_title('Per-Sample Energy Consumption')
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        # --- 右下：脉冲数 vs 总能耗散点 ---
        ax = axes[1, 1]
        ax.scatter(spikes, e_tot / 1e3, s=4, alpha=0.5, edgecolors='none')
        ax.set_xlabel('Spike Count')
        ax.set_ylabel('Total Energy (nJ)')
        ax.set_title('Spikes vs Energy')
        ax.grid(True, linewidth=0.3, alpha=0.5)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Power analysis figure saved to: {save_path}")

        plt.show()

    def plot_ieee_bootstrap_ci(self, metrics, save_path=None):
        """
        IEEE 风格的 Bootstrap 95% CI 图：
          左：误差条柱状图（4 项指标点估计 + CI 上下界）
          右：Bootstrap 重采样分布直方图（4 项指标叠加）
        """
        ci_labels = ['Accuracy', 'Macro F1', 'Sensitivity', 'Specificity']
        point_est = [metrics['accuracy_all'], metrics['f1_macro'],
                     metrics['sensitivity'],  metrics['specificity']]
        ci_pairs  = [metrics['bootstrap_acc_ci'],  metrics['bootstrap_f1_ci'],
                     metrics['bootstrap_sens_ci'],  metrics['bootstrap_spec_ci']]
        bs        = metrics['bootstrap_samples']
        boot_data = [bs['acc'], bs['f1'], bs['sens'], bs['spec']]
        colors    = ['#2166ac', '#4dac26', '#d6604d', '#762a83']

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        fig.suptitle('Bootstrap 95% Confidence Intervals (B=2000)', fontsize=10)

        # --- 左：误差条柱状图 ---
        ax = axes[0]
        x_pos   = np.arange(len(ci_labels))
        lo_errs = [pe - ci[0] for pe, ci in zip(point_est, ci_pairs)]
        hi_errs = [ci[1] - pe  for pe, ci in zip(point_est, ci_pairs)]
        bars = ax.bar(x_pos, point_est, width=0.5,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.errorbar(x_pos, point_est,
                    yerr=[lo_errs, hi_errs],
                    fmt='none', color='black', capsize=4,
                    capthick=0.8, elinewidth=0.8)
        # 柱顶标注点估计值和 CI
        for x, pe, ci in zip(x_pos, point_est, ci_pairs):
            ax.text(x, max(point_est) + max(hi_errs) + 0.04,
                    f'{pe:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]',
                    ha='center', va='bottom', fontsize=6, color='#222222')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ci_labels, fontsize=7.5)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.25)
        ax.set_title('Point Estimate + 95% CI')
        ax.grid(True, linewidth=0.3, alpha=0.5, axis='y')

        # --- 右：Bootstrap 分布直方图 ---
        ax2 = axes[1]
        for bd, pe, label, color in zip(boot_data, point_est, ci_labels, colors):
            bd_clean = bd[~np.isnan(bd)]
            ax2.hist(bd_clean, bins=40, alpha=0.55, color=color,
                     edgecolor='none', label=label, density=True)
            ax2.axvline(pe, color=color, linewidth=1.0, linestyle='--')
        ax2.set_xlabel('Metric Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Bootstrap Sampling Distributions')
        ax2.legend(fontsize=6.5, framealpha=0.7)
        ax2.grid(True, linewidth=0.3, alpha=0.5)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Bootstrap CI figure saved to: {save_path}")
        plt.show()

    def save_results(self, metrics):
        """保存测试结果（含能耗摘要）"""
        os.makedirs("test_results", exist_ok=True)
        results_file = f'test_results/{self.args.plot_name}_metrics.txt'
        with open(results_file, 'w') as f:
            f.write("Binary Classification Test Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Confusion Matrix:              {metrics['confusion_matrix']}\n")
            f.write(f"All activity accuracy:         {metrics['accuracy_all']:.4f}\n")
            f.write(f"Proportion weighting accuracy: {metrics['accuracy_proportion']:.4f}\n")
            f.write(f"Macro F1:                      {metrics['f1_macro']:.4f}\n")
            f.write(f"Sensitivity:                   {metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity:                   {metrics['specificity']:.4f}\n")
            f.write("\n--- Inference Time ---\n")
            f.write(f"Mean latency:  {metrics['inference_time_ms'].mean():.3f} ms\n")
            f.write(f"Std  latency:  {metrics['inference_time_ms'].std():.3f} ms\n")
            f.write("\n--- Energy Estimation ---\n")
            f.write(f"E_syn  mean:   {metrics['energy_syn_pj'].mean():.1f} pJ\n")
            f.write(f"E_neu  mean:   {metrics['energy_neu_pj'].mean():.1f} pJ\n")
            f.write(f"E_mem  mean:   {metrics['energy_mem_pj'].mean():.1f} pJ\n")
            f.write(f"E_total mean:  {metrics['energy_total_pj'].mean():.1f} pJ  "
                    f"({metrics['energy_total_pj'].mean()/1000:.3f} nJ)\n")
            f.write(f"P_avg  :       {metrics['power_avg_uw']:.4f} µW\n")
            f.write("\n--- Bootstrap 95% Confidence Intervals (B=2000) ---\n")
            f.write(f"Accuracy    : {metrics['accuracy_all']:.4f}  "
                    f"[{metrics['bootstrap_acc_ci'][0]:.4f}, {metrics['bootstrap_acc_ci'][1]:.4f}]\n")
            f.write(f"Macro F1    : {metrics['f1_macro']:.4f}  "
                    f"[{metrics['bootstrap_f1_ci'][0]:.4f}, {metrics['bootstrap_f1_ci'][1]:.4f}]\n")
            f.write(f"Sensitivity : {metrics['sensitivity']:.4f}  "
                    f"[{metrics['bootstrap_sens_ci'][0]:.4f}, {metrics['bootstrap_sens_ci'][1]:.4f}]\n")
            f.write(f"Specificity : {metrics['specificity']:.4f}  "
                    f"[{metrics['bootstrap_spec_ci'][0]:.4f}, {metrics['bootstrap_spec_ci'][1]:.4f}]\n")
        print(f"\nResults saved to: {results_file}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--time",          type=int,   default=250)
    parser.add_argument("--dt",            type=int,   default=1.0)
    parser.add_argument("--gpu",           dest="gpu", action="store_true")
    parser.add_argument("--no-gpu",        dest="gpu", action="store_false")
    parser.add_argument("--input_neuron",  type=int,   default=900)
    parser.add_argument("--n_neurons",     type=int,   default=900)
    parser.add_argument("--n_classes",     type=int,   default=2)
    parser.add_argument("--cuda_name",     type=str,   default='cuda:0')
    parser.add_argument("--plot_name",     type=str,
                        default='CIR_50_train_3w_gamma_seq_encode_real_raw_900_1_900_epoch_5')
    parser.add_argument("--lsm_out_data",  type=str,
                        default='Liquid_Out/CIR_50_test_3w_gamma_seq_encode_real_raw_900_1.pt')
    parser.add_argument("--tsne",          dest="tsne",               action="store_true")
    parser.add_argument("--no-tsne",       dest="tsne",               action="store_false")
    parser.add_argument("--confusion-matrix",    dest="save_confusion_matrix", action="store_true")
    parser.add_argument("--no-confusion-matrix", dest="save_confusion_matrix", action="store_false")
    parser.add_argument("--task",          type=str,   default='uwb')

    parser.set_defaults(gpu=True, tsne=False, save_confusion_matrix=False)
    args = parser.parse_args()

    tester = BinaryClassificationTester(args)
    tester.load_network()
    dataloader = tester.load_data()
    assignments, proportions = tester.load_assignments()

    n_test = len(dataloader)
    tester.run_testing(dataloader, assignments, proportions)
    metrics = tester.compute_metrics(n_test)

    # 混淆矩阵
    cm_path = f'confusion_matrix/{args.plot_name}_{metrics["accuracy_all"]:.4f}.svg'
    tester.plot_ieee_confusion_matrix(metrics, cm_path)

    # 分类指标对比
    cmp_path = f'confusion_matrix/{args.plot_name}_metrics_comparison.svg'
    tester.plot_ieee_metrics_comparison(metrics, cmp_path)

    # 能耗分析图（新增）
    pwr_path = f'confusion_matrix/{args.plot_name}_power_analysis.svg'
    tester.plot_ieee_power_analysis(metrics, pwr_path)

    # Bootstrap 置信区间图
    ci_path = f'confusion_matrix/{args.plot_name}_bootstrap_ci.svg'
    tester.plot_ieee_bootstrap_ci(metrics, ci_path)

    tester.save_results(metrics)


if __name__ == "__main__":
    main()