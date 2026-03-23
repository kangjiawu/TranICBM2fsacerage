#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNE-Python 批处理脚本：对 BIDS 格式 EEG 数据进行预处理和频谱分析
适用数据：EDF 文件 + events.tsv (BIDS 标准)
输出：每个被试每个 session 的五个频段功率 (delta, theta, alpha, beta, gamma)
"""

import os
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids  # 需要安装 mne-bids
import warnings

warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
bids_root = r'E:\openneurodata\ds003195-master'  # BIDS 根目录
output_dir = os.path.join(bids_root, 'derivatives', 'bandpower')
os.makedirs(output_dir, exist_ok=True)

# 预处理参数
notch_freq = 50  # 陷波频率 (Hz)，根据当地工频干扰设置
l_freq = 0.5  # 高通滤波
h_freq = 45  # 低通滤波（保留 gamma 需 >30，这里设为 45）
reject_criteria = dict(eeg=150e-6)  # 伪迹剔除阈值 (V)，150 µV，根据数据调整
tmin, tmax = -1, 2  # epoch 时间窗口 (s)，相对于事件
event_id = {'stimulus': 1}  # 需要提取的事件类型及其对应 ID（可根据 events.tsv 调整）
baseline = (None, 0)  # 基线校正时段 (None 表示不校正，或 (tmin, tmax))

# 频带定义 (Hz)
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# 存储所有结果的列表
all_results = []

# ==================== 获取所有被试 ====================
# 手动扫描 BIDS 目录，寻找 sub-* 文件夹
sub_dirs = [d for d in os.listdir(bids_root) if d.startswith('sub-')]
subjects = [d.replace('sub-', '') for d in sub_dirs]

print(f'找到被试: {subjects}')

# ==================== 主循环 ====================
for sub in subjects:
    sub_path = os.path.join(bids_root, f'sub-{sub}')

    # 查找该被试下的所有 session 文件夹
    ses_dirs = [d for d in os.listdir(sub_path) if d.startswith('ses-')]
    if not ses_dirs:
        # 如果没有 session，将整个被试视为一个 session
        ses_dirs = ['']

    for ses in ses_dirs:
        ses_id = ses.replace('ses-', '') if ses else ''
        print(f'\n========== 处理被试: {sub}, session: {ses_id} ==========')

        # 构建 eeg 文件夹路径
        if ses:
            eeg_dir = os.path.join(sub_path, ses, 'eeg')
        else:
            eeg_dir = os.path.join(sub_path, 'eeg')

        if not os.path.exists(eeg_dir):
            print(f'  跳过: EEG 文件夹不存在 {eeg_dir}')
            continue

        # 查找该 session 下的 EDF 文件
        edf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.edf')]
        if not edf_files:
            print(f'  跳过: 未找到 EDF 文件')
            continue

        # 通常一个 session 只有一个 EDF 文件，取第一个
        edf_fname = edf_files[0]
        edf_path = os.path.join(eeg_dir, edf_fname)

        # 查找对应的 events.tsv 文件
        events_fname = edf_fname.replace('_eeg.edf', '_events.tsv')
        events_path = os.path.join(eeg_dir, events_fname)
        if not os.path.exists(events_path):
            print(f'  警告: 未找到 events.tsv 文件 {events_path}，跳过')
            continue

        # ========== 1. 读取原始数据 ==========
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f'  错误: 无法读取 EDF 文件 {edf_path} - {e}')
            continue

        # 设置电极类型（假设全部为 EEG）
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

        # 加载 events.tsv 文件
        events_df = pd.read_csv(events_path, sep='\t')

        # 根据 events.tsv 创建 MNE 事件数组
        # 假设 events.tsv 包含 'onset' (秒), 'duration', 'trial_type' 列
        # 需要将 trial_type 映射为整数 ID
        unique_types = events_df['trial_type'].unique()
        type_to_id = {typ: idx + 1 for idx, typ in enumerate(unique_types)}  # 从 1 开始编号

        # 构建事件数组：每一行 [onset_samples, duration_samples, event_id]
        events = []
        for _, row in events_df.iterrows():
            onset_samp = int(round(row['onset'] * raw.info['sfreq']))
            dur_samp = int(round(row['duration'] * raw.info['sfreq'])) if not pd.isna(row['duration']) else 0
            event_code = type_to_id[row['trial_type']]
            events.append([onset_samp, dur_samp, event_code])
        events = np.array(events)

        # ========== 2. 预处理 ==========
        # 2.1 陷波滤波 (50 Hz)
        if notch_freq:
            raw.notch_filter(notch_freq, verbose=False)

        # 2.2 带通滤波
        raw.filter(l_freq, h_freq, verbose=False)

        # 2.3 坏道识别（简单方法：基于标准差，或从 channels.tsv 读取）
        # 这里演示从 channels.tsv 读取坏道标记（如果有）
        channels_fname = edf_fname.replace('_eeg.edf', '_channels.tsv')
        channels_path = os.path.join(eeg_dir, channels_fname)
        bads = []
        if os.path.exists(channels_path):
            chans_df = pd.read_csv(channels_path, sep='\t')
            # 假设 channels.tsv 有 'name' 和 'status' 列，status 为 'bad' 表示坏道
            if 'status' in chans_df.columns:
                bads = chans_df.loc[chans_df['status'] == 'bad', 'name'].tolist()
                raw.info['bads'] = bads
                print(f'    从 channels.tsv 标记坏道: {bads}')

        # 2.4 插值坏道（如果存在坏道）
        if bads:
            raw.interpolate_bads(reset_bads=True, verbose=False)

        # 2.5 重参考（平均参考）
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # ========== 3. 提取 Epochs ==========
        # 仅提取我们感兴趣的事件类型
        if event_id:
            # 检查事件类型是否在映射中
            selected_types = list(event_id.keys())
            for typ in selected_types:
                if typ not in type_to_id:
                    print(f'  警告: 事件类型 "{typ}" 不存在，可用类型: {list(type_to_id.keys())}')
                    # 移除不存在的类型
                    event_id.pop(typ, None)
            if not event_id:
                print('  跳过: 无有效事件类型')
                continue
        else:
            # 如果没有指定 event_id，使用所有事件类型
            event_id = type_to_id

        # 提取 epochs
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, verbose=False,
                            reject=reject_criteria, reject_by_annotation=False)

        if len(epochs) == 0:
            print('  跳过: 没有剩余的 epochs')
            continue

        # 可选：再次剔除坏 epoch（使用自动阈值）
        epochs.drop_bad(reject=reject_criteria, verbose=False)
        print(f'    保留 {len(epochs)} 个 epochs')

        # ========== 4. 计算频谱功率 ==========
        # 计算功率谱密度 (使用 Welch 方法)
        psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=0.5, fmax=45,
                                                   n_fft=256, n_overlap=128,
                                                   verbose=False)
        # psds 形状: (n_epochs, n_channels, n_freqs)

        # 提取每个频带的平均功率（对所有通道和 epochs 平均）
        # 也可以保留每个通道的功率，这里先平均
        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(freq_mask):
                print(f'  警告: 频带 {band} 无频率点')
                band_powers[band] = np.nan
                continue
            # 对频率点平均 -> 形状 (n_epochs, n_channels)
            psd_band = psds[:, :, freq_mask].mean(axis=2)
            # 再对 epochs 和 channels 平均，得到单个数值
            band_powers[band] = psd_band.mean()

        # 记录结果
        result = {'subject': sub, 'session': ses_id}
        result.update(band_powers)
        all_results.append(result)

        # 可选：保存预处理后的 epochs 以供后续分析
        # epochs_fname = f'sub-{sub}_ses-{ses_id}_epo.fif'
        # epochs.save(os.path.join(output_dir, epochs_fname), overwrite=True)

# ==================== 保存汇总结果 ====================
df = pd.DataFrame(all_results)
csv_path = os.path.join(output_dir, 'bandpower_summary.csv')
df.to_csv(csv_path, index=False)
print(f'\n所有处理完成！结果已保存至: {csv_path}')