import os
import torch
import numpy as np
import sys
import math
import random
from datetime import datetime  # 현재 시간 가져오기 위한 모듈 추가

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import griddata


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

fs = 48000.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

UNIT_DEGREE = 3
REF_0 = int(180/UNIT_DEGREE*0/180)
REF_60 = int(180/UNIT_DEGREE*60/180)
REF_63 = int(180/UNIT_DEGREE*63/180)     
REF_66 = int(180/UNIT_DEGREE*66/180)      
REF_69 = int(180/UNIT_DEGREE*69/180)      
REF_72 = int(180/UNIT_DEGREE*72/180)    
REF_75 = int(180/UNIT_DEGREE*75/180)
REF_78 = int(180/UNIT_DEGREE*78/180)
REF_84 = int(180/UNIT_DEGREE*84/180)
REF_90 = int(180/UNIT_DEGREE*90/180)     
REF_180 = int(180/UNIT_DEGREE*180/180)  

CONST_OBJ_DIR_LOW = REF_60
CONST_OBJ_DIR_HIGH = REF_180
CONST_OBJ_FREQ_LOW = 100.0
CONST_OBJ_FREQ_HIGH = 800.0

CONST_SPL_MIN_GOAL = -30
NUM_PARTICLES = 1024*3
NUM_ITERATIONS = 1000

def load_data():
    """
    데이터를 로드하여 4차원 텐서로 반환합니다.
    텐서의 형태: [num_drivers, num_angles, num_freq, 3] 
    (마지막 차원은 [freq, spl, phase]를 의미)
    
    Returns:
        torch.Tensor: 로드된 데이터의 4차원 텐서
    """
    # 하드코딩된 드라이버 디렉토리 목록
    driver_dirs = ['d1', 'd2', 'd3', 'd4']
    num_drivers = len(driver_dirs)
    
    all_drivers_data = []
    angle_set = set()
    freq_set = set()
    
    # 첫 번째 패스: 각도와 주파수의 동적 크기 결정
    for driver_dir in driver_dirs:
        if not os.path.isdir(driver_dir):
            raise FileNotFoundError(f"디렉토리 {driver_dir}이 존재하지 않습니다.")
        
        # 해당 드라이버 디렉토리 내의 hor *.txt 파일 목록
        files = [f for f in os.listdir(driver_dir) if f.startswith('hor') and f.endswith('.txt')]
        angles = sorted([int(f.split()[1].split('.')[0]) for f in files])
        angle_set.update(angles)
        
        for file in files:
            file_path = os.path.join(driver_dir, file)
            with open(file_path, 'r', encoding='cp949') as f:
                lines = f.readlines()[15:]  # 첫 15줄 무시
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            freq = float(parts[0])
                            freq_set.add(freq)
                        except ValueError:
                            continue  # 숫자로 변환할 수 없는 값은 무시
    
    # 정렬된 각도와 주파수 목록
    sorted_angles = sorted(angle_set)
    sorted_freqs = sorted(freq_set)
    num_angles = len(sorted_angles)
    num_freq = len(sorted_freqs)
    
    # 주파수 인덱스를 위한 매핑
    freq_to_idx = {freq: idx for idx, freq in enumerate(sorted_freqs)}
    
    # 데이터 초기화: [num_drivers, num_angles, num_freq, 3]
    data_tensor = torch.zeros((num_drivers, num_angles, num_freq, 3), dtype=torch.float32)
    
    # 드라이버별 데이터 로드
    for driver_idx, driver_dir in enumerate(driver_dirs):
        # 해당 드라이버 디렉토리 내의 hor *.txt 파일 목록
        files = [f for f in os.listdir(driver_dir) if f.startswith('hor') and f.endswith('.txt')]
        
        # 파일을 각도 순으로 정렬
        sorted_files = sorted(files, key=lambda x: int(x.split()[1].split('.')[0]))
        
        for angle_idx, file in enumerate(sorted_files):
            angle = int(file.split()[1].split('.')[0])
            if angle not in sorted_angles:
                continue  # 예상치 못한 각도는 무시
            
            file_path = os.path.join(driver_dir, file)
            with open(file_path, 'r', encoding='cp949') as f:
                lines = f.readlines()[15:]  # 첫 15줄 무시
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue  # 데이터가 충분하지 않은 줄은 무시
                    try:
                        freq, spl, phase = map(float, parts[:3])
                    except ValueError:
                        continue  # 숫자로 변환할 수 없는 값은 무시
                    if freq in freq_to_idx:
                        freq_idx = freq_to_idx[freq]
                        data_tensor[driver_idx, angle_idx, freq_idx, :] = torch.tensor([freq, spl, phase])
    
    return data_tensor

def load_data_ver():
    """
    데이터를 로드하여 4차원 텐서로 반환합니다.
    텐서의 형태: [num_drivers, num_angles, num_freq, 3] 
    (마지막 차원은 [freq, spl, phase]를 의미)
    
    Returns:
        torch.Tensor: 로드된 데이터의 4차원 텐서
    """
    # 하드코딩된 드라이버 디렉토리 목록
    driver_dirs = ['d1', 'd2', 'd3', 'd4']
    num_drivers = len(driver_dirs)
    
    all_drivers_data = []
    angle_set = set()
    freq_set = set()
    
    # 첫 번째 패스: 각도와 주파수의 동적 크기 결정
    for driver_dir in driver_dirs:
        if not os.path.isdir(driver_dir):
            raise FileNotFoundError(f"디렉토리 {driver_dir}이 존재하지 않습니다.")
        
        # 해당 드라이버 디렉토리 내의 ver *.txt 파일 목록
        files = [f for f in os.listdir(driver_dir) if f.startswith('ver') and f.endswith('.txt')]
        angles = sorted([int(f.split()[1].split('.')[0]) for f in files])
        angle_set.update(angles)
        
        for file in files:
            file_path = os.path.join(driver_dir, file)
            with open(file_path, 'r', encoding='cp949') as f:
                lines = f.readlines()[15:]  # 첫 15줄 무시
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            freq = float(parts[0])
                            freq_set.add(freq)
                        except ValueError:
                            continue  # 숫자로 변환할 수 없는 값은 무시
    
    # 정렬된 각도와 주파수 목록
    sorted_angles = sorted(angle_set)
    sorted_freqs = sorted(freq_set)
    num_angles = len(sorted_angles)
    num_freq = len(sorted_freqs)
    
    # 주파수 인덱스를 위한 매핑
    freq_to_idx = {freq: idx for idx, freq in enumerate(sorted_freqs)}
    
    # 데이터 초기화: [num_drivers, num_angles, num_freq, 3]
    data_tensor = torch.zeros((num_drivers, num_angles, num_freq, 3), dtype=torch.float32)
    
    # 드라이버별 데이터 로드
    for driver_idx, driver_dir in enumerate(driver_dirs):
        # 해당 드라이버 디렉토리 내의 ver *.txt 파일 목록
        files = [f for f in os.listdir(driver_dir) if f.startswith('ver') and f.endswith('.txt')]
        
        # 파일을 각도 순으로 정렬
        sorted_files = sorted(files, key=lambda x: int(x.split()[1].split('.')[0]))
        
        for angle_idx, file in enumerate(sorted_files):
            angle = int(file.split()[1].split('.')[0])
            if angle not in sorted_angles:
                continue  # 예상치 못한 각도는 무시
            
            file_path = os.path.join(driver_dir, file)
            with open(file_path, 'r', encoding='cp949') as f:
                lines = f.readlines()[15:]  # 첫 15줄 무시
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue  # 데이터가 충분하지 않은 줄은 무시
                    try:
                        freq, spl, phase = map(float, parts[:3])
                    except ValueError:
                        continue  # 숫자로 변환할 수 없는 값은 무시
                    if freq in freq_to_idx:
                        freq_idx = freq_to_idx[freq]
                        data_tensor[driver_idx, angle_idx, freq_idx, :] = torch.tensor([freq, spl, phase])
    
    return data_tensor

def peak(batched_data, batch_params):
    """
    RBJ Parametric Peaking EQ를 batch 처리하여
    주파수 응답(진폭, 위상)을 SPL과 Phase에 반영해주는 함수.

    Args:
        batched_data (torch.Tensor): [batch_size, num_drivers, num_angles, num_freq, 3]
            - 마지막 차원 3은 [freq(Hz), spl(dB), phase(deg)]를 의미
            - phase는 -180~180 범위 (degree) 가정
        batch_params (torch.Tensor): [batch_size, 80]
            - 최초 75개만 사용. d1=5필터×3파라미터, d2=10×3, d3=10×3
        fs (float): 샘플링 주파수(Hz). default=48000

    Returns:
        torch.Tensor: 필터가 적용된 shape=[batch_size, num_drivers, num_angles, num_freq, 3]
                      (주파수, dB SPL, Phase(deg))
    """
    device = batched_data.device

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) shape 및 데이터 분리
    # ─────────────────────────────────────────────────────────────────────────────
    batch_size, num_drivers, num_angles, num_freq, _ = batched_data.shape

    # (1) 주파수축 (Hz): 모든 배치/드라이버/각도에서 동일하다고 가정
    freq = batched_data[0, 0, 0, :, 0]  # [num_freq]
    freq = freq.to(device)

    # (2) SPL(dB), Phase(deg)
    spl_orig = batched_data[..., 1]    # [batch_size, num_drivers, num_angles, num_freq]
    phase_orig = batched_data[..., 2]  # 위상(도)

    # (3) batch_params에서 최초 75개 필터 파라미터만 사용
    params = batch_params[:, :75]  # shape=[batch_size, 75]

    # d1, d2, d3 분할
    # - d1:  5 filters × 3 params = 15
    # - d2: 10 filters × 3 params = 30
    # - d3: 10 filters × 3 params = 30
    # → 총 75
    d1_params = params[:, :15].view(batch_size, 5, 3)
    d2_params = params[:, 15:45].view(batch_size, 10, 3)
    d3_params = params[:, 45:].view(batch_size, 10, 3)
    # d4는 없음

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) RBJ 피킹 EQ 응답(진폭/위상)을 계산하는 함수
    # ─────────────────────────────────────────────────────────────────────────────
    def compute_parametric_peak_response(Fc, Gain_dB, Q, freq, fs=48000.0):
        """
        RBJ Parametric Peaking EQ의 주파수 응답(진폭 + 위상)을 계산한다.
        - Fc:  중심 주파수(Hz)
        - Gain_dB: 피킹 Gain (dB)
        - Q:  Q-factor
        - freq: 계산할 주파수 점들(Hz)
        - fs: 샘플링 주파수(Hz)
        
        Returns:
            (H_amp_dB, H_phase_deg)
            - H_amp_dB  : shape=[batch_size, num_filters, num_freq], dB 스케일
            - H_phase_deg: shape=[batch_size, num_filters, num_freq], 도(deg)
        """
        # (1) 주파수축 브로드캐스트 준비
        # freq.shape = [num_freq] -> [1, 1, num_freq]
        freq = freq.view(1, 1, -1)  # for broadcasting
        num_freq_local = freq.shape[2]

        # (2) 디지털 각주파수
        w = 2.0 * math.pi * freq / fs  # [1,1,num_freq]

        # (3) Gain_dB -> A = 10^(Gain_dB/40)
        A = torch.clamp(10.0 ** (Gain_dB / 40.0), min=1e-6)


        # (4) 중심 주파수 Fc -> w0
        epsilon_fc = 1e-3  # 최소 주파수 (Hz)
        Fc = torch.clamp(Fc, min=epsilon_fc)
        epsilon_q = 1e-6
        Q = torch.clamp(Q, min=epsilon_q)

        w0 = 2.0 * math.pi * Fc / fs  # [batch_size, num_filters, 1]
        alpha = torch.sin(w0) / (2.0 * Q)

        cos_w0 = torch.cos(w0)

        # (5) biquad 계수 계산 (배치 크기 유지)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A

        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        # (6) e^{-j w} = cos(w) - j sin(w), e^{-j 2w} = (e^{-j w})^2
        #    w.shape: [1,1,num_freq]
        cos_w = torch.cos(w)
        sin_w = torch.sin(w)
        z1 = torch.complex(cos_w, -sin_w)   # e^{-j w}
        z2 = z1 * z1              # e^{-j 2w}

        b0_c = torch.complex(b0, torch.zeros_like(b0))
        b1_c = torch.complex(b1, torch.zeros_like(b1))
        b2_c = torch.complex(b2, torch.zeros_like(b2))
        a0_c = torch.complex(a0, torch.zeros_like(a0))
        a1_c = torch.complex(a1, torch.zeros_like(a1))
        a2_c = torch.complex(a2, torch.zeros_like(a2))


        # (7) 분자/분모
        # b0,b1,b2,a0,a1,a2: [batch_size, num_filters, 1]
        # z1,z2: [1,1,num_freq]
        # => 브로드캐스트 후 최종 shape: [batch_size, num_filters, num_freq]
        num = b0_c + b1_c * z1 + b2_c * z2
        den = a0_c + a1_c * z1 + a2_c * z2
        H = num / den  # 복소 주파수 응답


        # (8) 진폭(dB), 위상(도) 계산
        #  - 진폭 = 20 log10(|H|)
        #  - 위상 = angle(H) [rad] -> deg
        H_abs = torch.abs(H)
        H_phase = torch.angle(H)
        H_amp_dB = 20.0 * torch.log10(H_abs + 1e-15)
        H_phase_deg = H_phase * 180.0 / math.pi


        return H_amp_dB, H_phase_deg

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) d1/d2/d3 각각 여러 필터(직렬 연결) → dB, 위상 각각 합산
    # ─────────────────────────────────────────────────────────────────────────────

    # -- d1
    Fc_d1   = d1_params[:, :, 0].unsqueeze(-1)  # [batch_size, 5, 1]
    Gain_d1 = d1_params[:, :, 1].unsqueeze(-1)
    Q_d1    = d1_params[:, :, 2].unsqueeze(-1)
    H_d1_dB, H_d1_deg = compute_parametric_peak_response(Fc_d1, Gain_d1, Q_d1, freq, fs=fs)
    # 직렬 연결 → 진폭(dB)는 합, 위상(도)도 합
    H_d1_amp_sum = H_d1_dB.sum(dim=1)   # [batch_size, num_freq]
    H_d1_phase_sum = H_d1_deg.sum(dim=1)

    # -- d2
    Fc_d2   = d2_params[:, :, 0].unsqueeze(-1)
    Gain_d2 = d2_params[:, :, 1].unsqueeze(-1)
    Q_d2    = d2_params[:, :, 2].unsqueeze(-1)
    H_d2_dB, H_d2_deg = compute_parametric_peak_response(Fc_d2, Gain_d2, Q_d2, freq, fs=fs)
    H_d2_amp_sum = H_d2_dB.sum(dim=1)
    H_d2_phase_sum = H_d2_deg.sum(dim=1)

    # -- d3
    Fc_d3   = d3_params[:, :, 0].unsqueeze(-1)
    Gain_d3 = d3_params[:, :, 1].unsqueeze(-1)
    Q_d3    = d3_params[:, :, 2].unsqueeze(-1)
    H_d3_dB, H_d3_deg = compute_parametric_peak_response(Fc_d3, Gain_d3, Q_d3, freq, fs=fs)
    H_d3_amp_sum = H_d3_dB.sum(dim=1)
    H_d3_phase_sum = H_d3_deg.sum(dim=1)

    # -- d4: 없음 → 0dB, 0deg
    H_d4_amp_sum = torch.zeros_like(H_d1_amp_sum)    # [batch_size, num_freq]
    H_d4_phase_sum = torch.zeros_like(H_d1_phase_sum)

    # -- batch_size, num_drivers, num_freq
    #    드라이버 순서: d1, d2, d3, d4
    H_amp_stack = torch.stack([H_d1_amp_sum, H_d2_amp_sum, H_d3_amp_sum, H_d4_amp_sum], dim=1)
    H_phase_stack = torch.stack([H_d1_phase_sum, H_d2_phase_sum, H_d3_phase_sum, H_d4_phase_sum], dim=1)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) 원본 SPL(dB), Phase(deg)에 결과를 적용
    # ─────────────────────────────────────────────────────────────────────────────
    # SPL : 원본 dB + 필터 dB (직렬 연결 → 합)
    # Phase: 원본 deg + 필터 deg (직렬 연결 → 합)

    # - 브로드캐스트 위해 [batch_size, num_drivers, 1, num_freq]
    H_amp_stack = H_amp_stack.unsqueeze(2)   # [batch_size, num_drivers, 1, num_freq]
    H_phase_stack = H_phase_stack.unsqueeze(2)

    spl_filtered = spl_orig + H_amp_stack
    phase_filtered = phase_orig + H_phase_stack

    # 위상 범위 -180~180 내로 정규화(optional)
    #   (phase + 180) % 360 - 180
    #   pytorch의 경우 float 텐서에 대해 % 연산이 그대로 되므로 아래와 같이 가능
    phase_filtered = (phase_filtered + 180.0) % 360.0 - 180.0

    # ─────────────────────────────────────────────────────────────────────────────
    # 5) 최종 텐서 조합
    # ─────────────────────────────────────────────────────────────────────────────
    # 마지막 차원: [freq(Hz), spl(dB), phase(deg)]
    batched_data_filtered = torch.stack([
        batched_data[..., 0],      # freq 그대로
        spl_filtered,              # 새 SPL
        phase_filtered             # 새 Phase
    ], dim=-1)

    return batched_data_filtered

def compute_1st_order_analog(freq: torch.Tensor,
                             fc: torch.Tensor,
                             filter_type: str = 'lpf'):
    """
    아날로그 1차(Butterworth) LPF/HPF 전달함수의 주파수 응답(진폭 dB, 위상 deg)을 계산한다.
    
    H_lpf(s) = ω0 / (s + ω0)
    H_hpf(s) = s   / (s + ω0)
    
    Args:
        freq (torch.Tensor): [num_freq], 주파수축(Hz)
        fc   (torch.Tensor): [batch_size, 1], 차단주파수(Hz)
        filter_type (str): 'lpf' 또는 'hpf'
    
    Returns:
        (amp_dB, phase_deg):
            amp_dB:    shape=[batch_size, num_freq]
            phase_deg: shape=[batch_size, num_freq]
    """
    # freq: [F]
    # fc  : [B, 1]
    # s   : [1, F]  (broadcast to [B, F])
    
    # 1) 각 배치별 ω0 = 2π fc
    omega_0 = 2.0 * math.pi * fc  # [B, 1]
    
    # 2) 주파수축 -> ω = 2π f
    w = 2.0 * math.pi * freq  # [F]
    
    # 3) 복소 주파수 s = jω
    #    shape 변환: w[None, :] => [1, F]
    #    broadcast => [B, F]
    s = 1j * w.unsqueeze(0)  # [1, F]  => broadcast with [B,1]
    
    # 4) 전달함수 계산
    if filter_type == 'lpf':
        # H_lpf(s) = ω0 / (s + ω0)
        # numerator   = ω0
        # denominator = s + ω0
        num = omega_0
        den = s + omega_0
    elif filter_type == 'hpf':
        # H_hpf(s) = s / (s + ω0)
        num = s
        den = s + omega_0
    else:
        raise ValueError("filter_type은 'lpf' 또는 'hpf'만 가능합니다.")
    
    H = num / den  # [B, F]
    
    # 5) 진폭(dB), 위상(deg)
    eps = 1e-15
    amp = torch.abs(H) + eps
    amp_dB = 20.0 * torch.log10(amp)  # [B, F]
    phase_rad = torch.angle(H)        # [B, F], rad
    phase_deg = phase_rad * (180.0 / math.pi)
    
    return amp_dB, phase_deg

def compute_2nd_order_analog(freq: torch.Tensor,
                             fc: torch.Tensor,
                             filter_type: str = 'lpf',
                             Q: float = 1.0/math.sqrt(2.0)):
    """
    아날로그 2차(Butterworth, Q=1/sqrt(2)) LPF/HPF 전달함수의 주파수 응답(진폭 dB, 위상 deg)을 계산한다.
    
    - 2차 Butterworth LPF (Q=1/√2):
        H_lpf(s) = ω0^2 / ( s^2 + √2 ω0 s + ω0^2 )
    - 2차 Butterworth HPF:
        H_hpf(s) = s^2   / ( s^2 + √2 ω0 s + ω0^2 )
    
    Args:
        freq (torch.Tensor): [num_freq], 주파수축(Hz)
        fc   (torch.Tensor): [batch_size, 1], 차단주파수(Hz)
        filter_type (str): 'lpf' 또는 'hpf'
        Q (float): Butterworth일 경우 보통 1/sqrt(2)
    
    Returns:
        (amp_dB, phase_deg):
            amp_dB:    shape=[batch_size, num_freq]
            phase_deg: shape=[batch_size, num_freq]
    """
    omega_0 = 2.0 * math.pi * fc  # [B, 1]
    w = 2.0 * math.pi * freq      # [F]
    s = 1j * w.unsqueeze(0)       # [B, F]
    
    # 기본적으로 2차 Butterworth면 damping=√2 이지만, Q=1/√2일 때 damping=√2가 맞음
    alpha = math.sqrt(2.0)  # = 1/Q for Q=1/sqrt(2)
    
    # s^2, s term
    s2 = s * s  # [B, F]
    
    if filter_type == 'lpf':
        # H_lpf(s) = ω0^2 / (s^2 + α ω0 s + ω0^2)
        num = omega_0**2
        den = s2 + alpha * omega_0 * s + omega_0**2
    elif filter_type == 'hpf':
        # H_hpf(s) = s^2 / (s^2 + α ω0 s + ω0^2)
        num = s2
        den = s2 + alpha * omega_0 * s + omega_0**2
    else:
        raise ValueError("filter_type은 'lpf' 또는 'hpf'만 가능합니다.")
    
    H = num / den  # [B, F]
    
    eps = 1e-15
    amp = torch.abs(H) + eps
    amp_dB = 20.0 * torch.log10(amp)
    phase_rad = torch.angle(H)
    phase_deg = phase_rad * (180.0 / math.pi)
    
    return amp_dB, phase_deg

def linkwitz_riley_analog(freq: torch.Tensor,
                          fc: torch.Tensor,
                          filter_type: str = 'lpf',
                          slope: int = 12):
    """
    아날로그 Linkwitz-Riley 크로스오버 (12dB 혹은 24dB/oct) 전달함수의
    주파수 응답(진폭 dB, 위상 deg)을 계산한다.
    
    - LR12: 1차 Butterworth × 1차 Butterworth
    - LR24: 2차 Butterworth × 2차 Butterworth
    
    Args:
        freq (torch.Tensor):  shape=[num_freq]
        fc   (torch.Tensor):  shape=[batch_size, 1]
        filter_type (str):    'lpf' 또는 'hpf'
        slope (int):          12 또는 24
        
    Returns:
        (amp_dB, phase_deg):  shape=[batch_size, num_freq]
    """
    if slope == 12:
        # 1차 버터워스 2개 직렬
        amp1, ph1 = compute_1st_order_analog(freq, fc, filter_type)
        amp2, ph2 = compute_1st_order_analog(freq, fc, filter_type)
        # 진폭(dB)는 합산, 위상(deg)도 합산
        amp_total = amp1 + amp2
        phase_total = ph1 + ph2
    elif slope == 24:
        # 2차 버터워스 2개 직렬
        amp1, ph1 = compute_2nd_order_analog(freq, fc, filter_type, Q=1.0/math.sqrt(2.0))
        amp2, ph2 = compute_2nd_order_analog(freq, fc, filter_type, Q=1.0/math.sqrt(2.0))
        amp_total = amp1 + amp2
        phase_total = ph1 + ph2
    else:
        raise ValueError("slope는 12 또는 24만 허용합니다.")
    
    return amp_total, phase_total

def crossover_analog(batched_data: torch.Tensor, batch_params: torch.Tensor):
    """
    아날로그 Linkwitz-Riley 크로스오버(12 or 24 dB/oct)를 적용하는 함수.
    
    - driver1: HPF (fc = batch_params[:, 75])
    - driver2: LPF (fc = batch_params[:, 76])
    - driver3: LPF (fc = batch_params[:, 77])
    - driver4: 필터 없음
    
    Args:
        batched_data (torch.Tensor): [B, 4, A, F, 3]  (freq, spl, phase)
        batch_params (torch.Tensor): [B, >=78]
        slope (int): 12 or 24 (dB/oct)
        
    Returns:
        out (torch.Tensor): [B, 4, A, F, 3]
            (freq, spl, phase) with analog crossover applied
    """
    device = batched_data.device
    B, D, A, F, _ = batched_data.shape
    if D != 4:
        raise ValueError("이 예제는 driver 4개인 경우만 가정합니다.")
    
    # 1) freq 축 가져오기 (모든 배치, 드라이버, 앵글 동일 가정)
    freq = batched_data[0, 0, 0, :, 0].to(device)  # shape=[F]
    
    # 2) SPL, Phase 원본
    spl_orig = batched_data[..., 1]   # [B, D, A, F]
    phase_orig = batched_data[..., 2] # [B, D, A, F]
    
    # 3) 차단주파수
    f_c1 = batch_params[:, 75].view(B, 1).to(device)  # driver1 HPF
    f_c2 = batch_params[:, 76].view(B, 1).to(device)  # driver2 LPF
    f_c3 = batch_params[:, 77].view(B, 1).to(device)  # driver3 LPF
    
    # 4) 각 드라이버별 아날로그 필터 계산
    # driver1: HPF
    amp_d1, phase_d1 = linkwitz_riley_analog(freq, f_c1, filter_type='hpf', slope=12)
    # driver2: LPF
    amp_d2, phase_d2 = linkwitz_riley_analog(freq, f_c2, filter_type='lpf', slope=12)
    # driver3: LPF
    amp_d3, phase_d3 = linkwitz_riley_analog(freq, f_c3, filter_type='lpf', slope=12)
    # driver4: 필터 없음 => 0dB, 0deg
    amp_d4 = torch.zeros_like(amp_d1)
    phase_d4 = torch.zeros_like(phase_d1)
    
    # angle 차원(A)을 위한 unsqueeze => shape [B, 1, F]
    amp_d1 = amp_d1.unsqueeze(1)   # [B, 1, F]
    amp_d2 = amp_d2.unsqueeze(1)
    amp_d3 = amp_d3.unsqueeze(1)
    amp_d4 = amp_d4.unsqueeze(1)
    ph_d1  = phase_d1.unsqueeze(1)
    ph_d2  = phase_d2.unsqueeze(1)
    ph_d3  = phase_d3.unsqueeze(1)
    ph_d4  = phase_d4.unsqueeze(1)
    
    # 최종 덧셈 (dB/deg)
    # 드라이버 순서는 (d1, d2, d3, d4)
    driver_amp_stack = torch.stack([amp_d1, amp_d2, amp_d3, amp_d4], dim=1)  # [B, 4, 1, F]
    driver_phase_stack = torch.stack([ph_d1, ph_d2, ph_d3, ph_d4], dim=1)    # [B, 4, 1, F]
    
    # SPL = 원본(dB) + 필터(dB), Phase = 원본(deg) + 필터(deg)
    spl_filtered = spl_orig + driver_amp_stack  # [B, 4, A, F]
    phase_filtered = phase_orig + driver_phase_stack
    
    # 위상 래핑 -180 ~ +180
    phase_filtered = (phase_filtered + 180.0) % 360.0 - 180.0
    
    # freq는 기존과 동일
    out = torch.stack([
        batched_data[..., 0],  # freq
        spl_filtered,
        phase_filtered
    ], dim=-1)  # [B, 4, A, F, 3]
    
    return out

def delay(data: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    data:   5D 텐서 [batch_size, num_drivers, num_angles, num_freq, 3]
            마지막 차원은 [freq(Hz), spl(dB), phase(°)]
    params: 2D 텐서 [batch_size, 80]
            여기서 마지막 2개 파라미터만 사용 (드라이버 2, 3의 delay[μs])
    
    Returns:
        지연(delay)만큼 위상이 보정된 data와 동일한 shape의 5D 텐서
    """
    
    # 배치 차원이 가장 앞에 있으므로, 이미 배치별 병렬처리가 가능한 형태임.
    # GPU에 올라가 있지 않다면 to(device)로 옮겨서 연산
    data = data.to(device)
    params = params.to(device)

    # 위상 wrapping 함수: [-180, 180] 범위로 정규화
    def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
        return (phase + 180.0) % 360.0 - 180.0

    # -----------------------------
    # 1) 두 번째 드라이버 (driver_idx = 1) 
    # -----------------------------
    # delay(μs) = params[..., 78]
    driver2_delay_us = params[:, 78].reshape(-1, 1, 1)  # shape [batch_size, 1, 1]
    
    # freq = data[:, 1, :, :, 0]
    # phase = data[:, 1, :, :, 2]
    freq_driver2 = data[:, 1, :, :, 0]
    phase_driver2 = data[:, 1, :, :, 2]

    # 위상 이동량(도 단위)
    phase_shift_d2 = -driver2_delay_us * freq_driver2 * 360e-6  # 360 * f * (delay * 1e-6)

    # 기존 위상에 더하고 wrap
    new_phase_driver2 = wrap_phase(phase_driver2 + phase_shift_d2)

    # 업데이트
    data[:, 1, :, :, 2] = new_phase_driver2

    # -----------------------------
    # 2) 세 번째 드라이버 (driver_idx = 2)
    # -----------------------------
    # delay(μs) = params[..., 79]
    driver3_delay_us = params[:, 79].reshape(-1, 1, 1)
    
    freq_driver3 = data[:, 2, :, :, 0]
    phase_driver3 = data[:, 2, :, :, 2]

    phase_shift_d3 = -driver3_delay_us * freq_driver3 * 360e-6
    new_phase_driver3 = wrap_phase(phase_driver3 + phase_shift_d3)

    data[:, 2, :, :, 2] = new_phase_driver3

    return data

def combine_and_normalize(data: torch.Tensor) -> torch.Tensor:
    """
    data: 5D Tensor 
          [batch_size, num_drivers=4, num_angles, num_freq, (freq, spl, phase)]
    return:
          5D Tensor
          [batch_size, 1, num_angles, num_freq, (freq, spl, phase)]
    """
    # 혹시 모르니 data가 GPU에 올라와 있지 않다면 to(device)로 복사
    data = data.to(device)
    
    # shape 파악
    batch_size, num_drivers, num_angles, num_freq, _ = data.shape
    
    # freq, spl, phase 각각을 분리 (모두 동일 shape: [batch_size, num_drivers, num_angles, num_freq])
    freq_data  = data[..., 0]  # [B, D, A, F]
    spl_data   = data[..., 1]  # [B, D, A, F]
    phase_data = data[..., 2]  # [B, D, A, F]

    # === 1) Combine (복소 합산) ===
    # amplitude = 10^(spl/20)
    amplitude = torch.pow(10.0, spl_data / 20.0)  # [B, D, A, F]
    
    # phase degree -> radian
    phase_rad = phase_data * math.pi / 180.0
    
    # real, imag 파트
    re = amplitude * torch.cos(phase_rad)  # [B, D, A, F]
    im = amplitude * torch.sin(phase_rad)  # [B, D, A, F]
    
    # driver 차원(dimension=1) 합산
    re_sum = re.sum(dim=1)  # [B, A, F]
    im_sum = im.sum(dim=1)  # [B, A, F]
    
    # amplitude, phase로 변환
    # amp_sum이 0이 되는 경우를 방지해 작은 epsilon 추가
    eps = 1e-12
    amp_sum = torch.sqrt(re_sum * re_sum + im_sum * im_sum + eps)  # [B, A, F]
    
    # 최종 spl (dB)
    spl_out = 20.0 * torch.log10(amp_sum + eps)  # [B, A, F]
    
    # 최종 phase (degree)
    phase_out = torch.atan2(im_sum, re_sum) * (180.0 / math.pi)  # [B, A, F]
    phase_out = (phase_out + 180.0) % 360.0 - 180.0
    # freq 값은 모든 driver가 동일한 freq 축을 갖는다고 가정하므로,
    # 그냥 첫 번째 driver의 freq를 사용 (혹은 mean 등). 여기서는 0번째 driver 사용
    # shape: [B, A, F]
    freq_out = freq_data[:, 0, :, :]  # [B, A, F]

    # === 2) Normalize ===
    # angle=0 인덱스의 SPL 값을 기준으로 모든 angle의 SPL을 뺌
    # angle=0 => index=0 (주의: 실제 각도가 0 deg가 아니라 "첫 번째 angle 인덱스"라는 의미)
    
    # shape 맞추기를 위해 unsqueeze
    # spl_out[:, 0, :] => [B, F]
    ref_spl = spl_out[:, 0, :]  # shape [B, F]
    # broadcasting을 위해 angle 축(=2) 추가
    ref_spl = ref_spl.unsqueeze(1)  # [B, 1, F]
    
    # spl_out에서 빼주기
    spl_out_norm = spl_out - ref_spl  # [B, A, F]

    # === 3) 최종 결과 텐서 만들기 ===
    # 출력 형태: [batch_size, 1, num_angles, num_freq, 3]
    # 마지막 차원: (freq, spl, phase)
    
    # freq_out: [B, A, F] => (freq_out.unsqueeze(1)) => [B, 1, A, F]
    # spl_out_norm: [B, A, F] => [B, 1, A, F]
    # phase_out: [B, A, F] => [B, 1, A, F]
    
    freq_out_5d   = freq_out.unsqueeze(1)   # [B, 1, A, F]
    spl_out_5d    = spl_out_norm.unsqueeze(1)  # [B, 1, A, F]
    phase_out_5d  = phase_out.unsqueeze(1)  # [B, 1, A, F]
    
    # 마지막 차원을 (freq, spl, phase) 로 stack
    # stack(dim=-1) => shape: [B, 1, A, F, 3]
    out = torch.stack([freq_out_5d, spl_out_5d, phase_out_5d], dim=-1)
    
    return out

def objective(batched_data_hor, batched_params, batched_data_ver):
    device = batched_data_hor.device
    batched_params = batched_params.to(device)
    batched_data_ver = batched_data_ver.to(device)

    # ─────────────────────────────────────────────────────────────────
    # 1) 우선 horizontal 데이터에 대해 기존 파이프라인 수행
    # ─────────────────────────────────────────────────────────────────
    # 1) Peak EQ 적용
    x_hor = peak(batched_data_hor, batched_params)
    # 2) Crossover 적용
    x_hor = crossover_analog(x_hor, batched_params)
    # 3) Delay 적용
    x_hor = delay(x_hor, batched_params)
    # 4) Combine & Normalize
    x_hor = combine_and_normalize(x_hor)  
    #    x_hor.shape = [B, 1, A, F, 3]

    # freq, spl 분리
    freq_hor = x_hor[..., 0]  # [B, 1, A, F]
    spl_hor  = x_hor[..., 1]  # [B, 1, A, F]

    # angle 80~180도 구간(REF_72:REF_180)
    spl_sub_h  = spl_hor[:, 0, CONST_OBJ_DIR_LOW:CONST_OBJ_DIR_HIGH, :]  
    freq_sub_h = freq_hor[:, 0, CONST_OBJ_DIR_LOW:CONST_OBJ_DIR_HIGH, :]

    # freq 100~1500 범위 마스킹
    mask_h = (freq_sub_h >= CONST_OBJ_FREQ_LOW) & (freq_sub_h <= CONST_OBJ_FREQ_HIGH)  
    mask_float_h = mask_h.float()

    spl_sum_h    = (spl_sub_h * mask_float_h).sum(dim=(1, 2))          
    mask_count_h = mask_float_h.sum(dim=(1, 2))                        
    cost_hor     = spl_sum_h / (mask_count_h + 1e-15)  # [B]

    # ─────────────────────────────────────────────────────────────────
    # 2) vertical 데이터도 동일 파이프라인 수행
    # ─────────────────────────────────────────────────────────────────
    x_ver = peak(batched_data_ver, batched_params)
    x_ver = crossover_analog(x_ver, batched_params)
    x_ver = delay(x_ver, batched_params)
    x_ver = combine_and_normalize(x_ver)
    # x_ver.shape = [B, 1, A, F, 3]

    freq_ver = x_ver[..., 0]  # [B, 1, A, F]
    spl_ver  = x_ver[..., 1]  # [B, 1, A, F]

    # angle 70~180도
    spl_sub_v  = spl_ver[:, 0, CONST_OBJ_DIR_LOW:CONST_OBJ_DIR_HIGH, :]
    freq_sub_v = freq_ver[:, 0, CONST_OBJ_DIR_LOW:CONST_OBJ_DIR_HIGH, :]

    # freq CONST_OBJ_FREQ_LOW to CONST_OBJ_FREQ_HIGH
    mask_v = (freq_sub_v >= CONST_OBJ_FREQ_LOW) & (freq_sub_v <= CONST_OBJ_FREQ_HIGH)
    mask_float_v = mask_v.float()

    spl_sum_v    = (spl_sub_v * mask_float_v).sum(dim=(1, 2))
    mask_count_v = mask_float_v.sum(dim=(1, 2))
    cost_ver     = spl_sum_v / (mask_count_v + 1e-15)  # [B]

    # ─────────────────────────────────────────────────────────────────
    # 3) Linstening Windows Cost hor
    # ─────────────────────────────────────────────────────────────────

    lw_spl_sub_h  = spl_hor[:, 0, REF_0:CONST_OBJ_DIR_LOW, :]  
    lw_freq_sub_h = freq_hor[:, 0, REF_0:CONST_OBJ_DIR_LOW, :]

    # freq 100~1500 범위 마스킹
    lw_mask_h = (lw_freq_sub_h >= CONST_OBJ_FREQ_LOW) & (lw_freq_sub_h <= CONST_OBJ_FREQ_HIGH)  
    lw_mask_float_h = lw_mask_h.float()

    lw_spl_sum_h    = (lw_spl_sub_h * lw_mask_float_h).sum(dim=(1, 2))          
    lw_mask_count_h = mask_float_h.sum(dim=(1, 2))                        
    lw_cost_hor     = lw_spl_sum_h / (lw_mask_count_h + 1e-15)  # [B]

    # ─────────────────────────────────────────────────────────────────
    # 4) Linstening Windows Cost ver
    # ─────────────────────────────────────────────────────────────────

    lw_spl_sub_v  = spl_ver[:, 0, REF_0:CONST_OBJ_DIR_LOW, :]  
    lw_freq_sub_v = freq_ver[:, 0, REF_0:CONST_OBJ_DIR_LOW, :]

    # freq 100~1500 범위 마스킹
    lw_mask_v = (lw_freq_sub_v >= CONST_OBJ_FREQ_LOW) & (lw_freq_sub_v <= CONST_OBJ_FREQ_HIGH)  
    lw_mask_float_v = lw_mask_v.float()

    lw_spl_sum_v    = (lw_spl_sub_v * lw_mask_float_v).sum(dim=(1, 2))          
    lw_mask_count_v = mask_float_v.sum(dim=(1, 2))                        
    lw_cost_ver     = lw_spl_sum_v / (lw_mask_count_v + 1e-15)  # [B]

    # ─────────────────────────────────────────────────────────────────
    # 5) Final Cost
    #    shape=[B], 맨 끝에 view(-1,1)로 만들어 [B,1]
    # ─────────────────────────────────────────────────────────────────
    cost = cost_hor + cost_ver - lw_cost_hor - lw_cost_ver
    cost = cost.view(-1, 1)  # [B, 1]

    return cost

class ParticleSwarmOptimizer:
    def __init__(
        self,
        data: torch.Tensor,
        num_particles: int = 32,
        max_iter: int = 50,
        w: float = 0.5,      # 관성 계수
        c1: float = 1.5,     # 파티클 자신의 최적점으로 끌어가는 계수
        c2: float = 1.5,     # 전역 최적점으로 끌어가는 계수
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            data (torch.Tensor): 4차원 텐서 [4, num_angles, num_freq, 3]
            num_particles (int): 파티클(개체) 수
            max_iter (int): 최대 반복 횟수
            w (float): 관성 계수
            c1 (float): 개인 최적 해로 끌리는 계수
            c2 (float): 전역 최적 해로 끌리는 계수
            device (torch.device): 연산을 수행할 디바이스
        """
        self.data = data.to(device)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.device = device

        # 각 파라미터 인덱스별로 min_val, max_val을 정의
        self.bounds = self._create_bounds().to(self.device)  # shape [80, 2]

        # 파라미터 초기화: [num_particles, 80]
        self.positions = self._init_positions()  # GPU 텐서
        self.velocities = self._init_velocities() # GPU 텐서

        # 개인 최적 해(pbest) 및 최적 cost
        self.pbest = self.positions.clone()
        self.pbest_cost = torch.full((num_particles, 1), float('inf'), device=self.device)

        # 전역 최적 해(gbest) 및 그 cost
        self.gbest = torch.zeros((80,), device=self.device)
        self.gbest_cost = float('inf')

    def _create_bounds(self):
        """
        문제에서 제시된 80개 파라미터의 [min, max] 범위를
        80 x 2 형태 텐서로 반환
        """
        bounds_list = []
        for i in range(15):
            if i % 3 == 0:
                bounds_list.append([100.0, 600.0])  # freq
            elif i % 3 == 1:
                bounds_list.append([-1.0, 1.0])      # gain
            else:
                bounds_list.append([1.0, 5.0])       # Q

        # 15 <= i <= 35
        for i in range(15, 36):
            if i % 3 == 0:
                bounds_list.append([100.0, 600.0])  # freq
            elif i % 3 == 1:
                bounds_list.append([-3.0, 3.0])      # gain
            else:
                bounds_list.append([1.0, 5.0])       # Q

        # 36 <= i <= 44
        for i in range(36, 45):
            if i % 3 == 0:
                bounds_list.append([600.0, 800.0])  # freq
            elif i % 3 == 1:
                bounds_list.append([-3.0, 3.0])      # gain
            else:
                bounds_list.append([5.0, 10.0])       # Q

        # 45 <= i <= 65
        for i in range(45, 66):
            if i % 3 == 0:
                bounds_list.append([100.0, 400.0])  # freq
            elif i % 3 == 1:
                bounds_list.append([-6.0, 6.0])      # gain
            else:
                bounds_list.append([1.0, 5.0])       # Q

        # 66 <= i <= 74
        for i in range(66, 75):
            if i % 3 == 0:
                bounds_list.append([400.0, 800.0])  # freq
            elif i % 3 == 1:
                bounds_list.append([-3.0, 3.0])      # gain
            else:
                bounds_list.append([5.0, 10.0])       # Q


        # 인덱스 75~79
        bounds_list.append([150.0, 300.0])   # index 75
        bounds_list.append([100.0, 1000.0])  # index 76
        bounds_list.append([100.0, 600.0])   # index 77
        bounds_list.append([0.0,   300.0])   # index 78
        bounds_list.append([0.0,   1000.0])  # index 79

        return torch.tensor(bounds_list, dtype=torch.float32)

    def _init_positions(self):
        """
        각 파라미터 범위 내에서 균일분포로 랜덤 초기화
        positions shape = [num_particles, 80]
        """
        num_params = self.bounds.shape[0]
        # shape: [1, 80]
        min_vals = self.bounds[:, 0].unsqueeze(0)
        max_vals = self.bounds[:, 1].unsqueeze(0)
        # 무작위 [0,1] 텐서
        rand_u = torch.rand(self.num_particles, num_params, device=self.device)
        # broadcasting으로 초기화
        positions = min_vals + (max_vals - min_vals) * rand_u
        return positions

    def _init_velocities(self):
        """
        초기 속도는 파라미터 범위의 10% 이내에서 무작위 설정(혹은 0으로 설정해도 무방)
        velocities shape = [num_particles, 80]
        """
        num_params = self.bounds.shape[0]
        min_vals = self.bounds[:, 0].unsqueeze(0)
        max_vals = self.bounds[:, 1].unsqueeze(0)

        # 범위의 10% => (max - min) * 0.1
        vel_range = (max_vals - min_vals) * 0.1
        rand_u = (2.0 * torch.rand(self.num_particles, num_params, device=self.device) - 1.0)
        # -0.1 ~ +0.1 배
        velocities = rand_u * vel_range
        return velocities

    def _clamp_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        문제에서 정의한 bounds 범위로 positions를 clamp
        shape: [num_particles, 80]
        """
        min_vals = self.bounds[:, 0].unsqueeze(0)  # [1, 80]
        max_vals = self.bounds[:, 1].unsqueeze(0)  # [1, 80]
        clamped = torch.max(torch.min(positions, max_vals), min_vals)
        return clamped

    def _clamp_velocities(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        속도를 너무 크게 방치하면 해가 발산할 가능성이 있으므로,
        문제에 따라 velocity clamp를 적용 가능.
        여기서는 위치 범위의 20%로 clamp 예시.
        """
        min_vals = self.bounds[:, 0].unsqueeze(0)
        max_vals = self.bounds[:, 1].unsqueeze(0)
        vel_limit = (max_vals - min_vals) * 0.2

        # vel_limit는 양수이므로, -vel_limit ~ +vel_limit로 clamp
        clamped = torch.max(-vel_limit, torch.min(velocities, vel_limit))
        return clamped

    def optimize(self, objective_func, batched_data_ver):
        """
        PSO 메인 루프 실행 (수정본)
        Args:
            objective_func (callable): objective 함수
                                    호출 시 shape=[num_particles, 1] 반환
            batched_data_ver (Tensor): 검증용 데이터
        """

        # ---------------------------------------------------------------------
        # PSO를 "여러 번" 돌려서 목표를 여러 번 달성하기 위해, 
        # 전체 과정을 무한 루프로 감쌉니다.
        # ---------------------------------------------------------------------
        # data를 [num_particles, 4, A, F, 3]로 확장 (batch dimension 추가)
        batched_data = self.data.unsqueeze(0).expand(self.num_particles, -1, -1, -1, -1).contiguous()
        batched_data_ver = batched_data_ver.unsqueeze(0).expand(self.num_particles, -1, -1, -1, -1).contiguous()
        reset_interval = self.max_iter // 5

        total_iterations = 0

        while True:
            # ------------------
            # 0) 초기화 & reset_interval 단위 반복
            # ------------------
            while True:
                print(f"ititialize. total_iter : {total_iterations}")
                # 파티클 초기화
                self.positions = self._init_positions()
                self.velocities = self._init_velocities()

                # 개인 최적 (pbest) 초기화
                self.pbest = self.positions.clone()
                self.pbest_cost = torch.full((self.num_particles, 1), float('inf'), device=self.device)

                # 전역 최적 (gbest) 초기화
                self.gbest = torch.zeros((self.bounds.shape[0],), device=self.device)
                self.gbest_cost = float('inf')

                # ------------------
                # 1) reset_interval 반복 수행
                # ------------------
                for iteration in range(reset_interval):
                    current_iter = total_iterations + 1
                    # 1) Objective 계산
                    costs = objective_func(batched_data, self.positions, batched_data_ver)

                    # 2) 개인 최적 갱신
                    update_mask = costs < self.pbest_cost
                    self.pbest[update_mask.squeeze()] = self.positions[update_mask.squeeze()]
                    self.pbest_cost[update_mask] = costs[update_mask]

                    # 3) 전역 최적 갱신
                    min_cost, min_idx = torch.min(costs, dim=0)
                    min_idx = min_idx.item()
                    if min_cost.item() < self.gbest_cost:
                        self.gbest_cost = min_cost.item()
                        self.gbest = self.positions[min_idx].clone()

                    # 4) 속도, 위치 업데이트
                    r1 = torch.rand_like(self.positions)
                    r2 = torch.rand_like(self.positions)

                    inertia = self.w * self.velocities
                    cognitive = self.c1 * r1 * (self.pbest - self.positions)
                    social = self.c2 * r2 * (self.gbest.unsqueeze(0) - self.positions)

                    self.velocities = inertia + cognitive + social
                    self.velocities = self._clamp_velocities(self.velocities)  # 속도 clamp
                    self.positions = self.positions + self.velocities
                    self.positions = self._clamp_positions(self.positions)      # 위치 clamp

                    # 디버깅 출력
                    if current_iter % (self.max_iter // 10) == 0:
                        print(f"Iter[{current_iter}/{self.max_iter}] gbest_cost: {self.gbest_cost:.6f}")

                    total_iterations += 1

                # reset_interval 만큼 진행 후 목표 달성 여부 확인
                print(f"after {reset_interval} iteration, gbest_cost: {self.gbest_cost:.6f}")
                if self.gbest_cost <= CONST_SPL_MIN_GOAL:
                    print("goal meet. continue rest.")
                    break  # reset_interval 반복을 탈출 -> 나머지 반복 수행
                else:
                    print("goal meet failed. reset.\n")
                    # 목표 실패 시 다시 reset_interval 반복(초기화)으로 돌아감
                    continue

            # ------------------
            # 2) 나머지 반복 수행
            # ------------------
            remaining_iters = self.max_iter - reset_interval
            for iteration in range(remaining_iters):
                current_iter = total_iterations + 1
                # 1) Objective 계산
                costs = objective_func(batched_data, self.positions, batched_data_ver)

                # 2) 개인 최적 갱신
                update_mask = costs < self.pbest_cost
                self.pbest[update_mask.squeeze()] = self.positions[update_mask.squeeze()]
                self.pbest_cost[update_mask] = costs[update_mask]

                # 3) 전역 최적 갱신
                min_cost, min_idx = torch.min(costs, dim=0)
                min_idx = min_idx.item()
                if min_cost.item() < self.gbest_cost:
                    self.gbest_cost = min_cost.item()
                    self.gbest = self.positions[min_idx].clone()

                # 4) 속도, 위치 업데이트
                r1 = torch.rand_like(self.positions)
                r2 = torch.rand_like(self.positions)

                inertia = self.w * self.velocities
                cognitive = self.c1 * r1 * (self.pbest - self.positions)
                social = self.c2 * r2 * (self.gbest.unsqueeze(0) - self.positions)

                self.velocities = inertia + cognitive + social
                self.velocities = self._clamp_velocities(self.velocities)
                self.positions = self.positions + self.velocities
                self.positions = self._clamp_positions(self.positions)

                # 디버깅 출력
                if current_iter % (self.max_iter // 10) == 0:
                    print(f"Iter[{current_iter}/{self.max_iter}] gbest_cost: {self.gbest_cost:.6f}")

                total_iterations += 1

            # ------------------
            # 3) 한 번의 PSO 루프(= max_iter) 완료 후 결과 출력
            # ------------------
            print(f"final gbest_cost: {self.gbest_cost:.6f} (total_iteration: {total_iterations})")

            # main 함수에서처럼 결과를 동일한 포맷으로 출력
            best_params = self.gbest
            best_cost = self.gbest_cost

            # 결과를 파일로 저장하기 위한 준비
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 예: 20250114_123456
            filename = f"pso_result_{current_time}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== PSO 최적화 결과 ===\n")
                f.write(f"Best Cost : {best_cost:.4f}\n\n")

                f.write("D1 필터 파라미터 (5개):\n")
                for i in range(5):
                    fc   = best_params[i*3 + 0]
                    gain = best_params[i*3 + 1]
                    qval = best_params[i*3 + 2]
                    f.write(f"  Filter {i+1}: Freq = {fc:.2f} Hz, Gain = {gain:.2f} dB, Q = {qval:.2f}\n")
                f.write(f"D1 HPF crossover freq: {best_params[75]:.2f} Hz\n\n")

                f.write("D2 필터 파라미터 (10개):\n")
                for i in range(10):
                    fc   = best_params[15 + i*3 + 0]
                    gain = best_params[15 + i*3 + 1]
                    qval = best_params[15 + i*3 + 2]
                    f.write(f"  Filter {i+1}: Freq = {fc:.2f} Hz, Gain = {gain:.2f} dB, Q = {qval:.2f}\n")
                f.write(f"D2 LPF crossover freq: {best_params[76]:.2f} Hz\n")
                f.write(f"D2 Delay: {best_params[78]:.2f} us\n\n")

                f.write("D3 필터 파라미터 (10개):\n")
                for i in range(10):
                    fc   = best_params[45 + i*3 + 0]
                    gain = best_params[45 + i*3 + 1]
                    qval = best_params[45 + i*3 + 2]
                    f.write(f"  Filter {i+1}: Freq = {fc:.2f} Hz, Gain = {gain:.2f} dB, Q = {qval:.2f}\n")
                f.write(f"D3 LPF crossover freq: {best_params[77]:.2f} Hz\n")
                f.write(f"D3 Delay: {best_params[79]:.2f} us\n")
            print(f"=== PSO 최적화 결과가 '{filename}' 파일에 저장되었습니다 ===")

            # ---------------------------------------------------------------------
            # 기존에는 여기서 return 하였으나,
            # "프로그램을 종료하지 않고 다시 처음부터 무한히 돌기" 위해 return을 제거하고,
            # while 루프의 처음으로 돌아갑니다.
            # ---------------------------------------------------------------------
            # continue를 만나면 while True 가장 바깥 루프 맨 위로 돌아가 새로 초기화
            continue
    
def peak_p(batched_data, batch_params):

    device = batched_data.device

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) shape 및 데이터 분리
    # ─────────────────────────────────────────────────────────────────────────────
    batch_size, num_drivers, num_angles, num_freq, _ = batched_data.shape

    # (1) 주파수축 (Hz): 모든 배치/드라이버/각도에서 동일하다고 가정
    freq = batched_data[0, 0, 0, :, 0]  # [num_freq]
    freq = freq.to(device)

    # (2) SPL(dB), Phase(deg)
    spl_orig = batched_data[..., 1]    # [batch_size, num_drivers, num_angles, num_freq]
    phase_orig = batched_data[..., 2]  # 위상(도)

    # (3) batch_params에서 최초 48개 필터 파라미터만 사용
    params = batch_params[:, :48]  # shape=[batch_size, 75]

    d4_params = params[:, :24].view(batch_size, 8, 3)
    d1_params = params[:, 24:36].view(batch_size, 4, 3)
    d2_params = params[:, 36:42].view(batch_size, 2, 3)
    d3_params = params[:, 42:48].view(batch_size, 2, 3)

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) RBJ 피킹 EQ 응답(진폭/위상)을 계산하는 함수
    # ─────────────────────────────────────────────────────────────────────────────
    def compute_parametric_peak_response(Fc, Gain_dB, Q, freq, fs=48000.0):
        """
        RBJ Parametric Peaking EQ의 주파수 응답(진폭 + 위상)을 계산한다.
        - Fc:  중심 주파수(Hz)
        - Gain_dB: 피킹 Gain (dB)
        - Q:  Q-factor
        - freq: 계산할 주파수 점들(Hz)
        - fs: 샘플링 주파수(Hz)
        
        Returns:
            (H_amp_dB, H_phase_deg)
            - H_amp_dB  : shape=[batch_size, num_filters, num_freq], dB 스케일
            - H_phase_deg: shape=[batch_size, num_filters, num_freq], 도(deg)
        """
        # (1) 주파수축 브로드캐스트 준비
        # freq.shape = [num_freq] -> [1, 1, num_freq]
        freq = freq.view(1, 1, -1)  # for broadcasting
        num_freq_local = freq.shape[2]

        # (2) 디지털 각주파수
        w = 2.0 * math.pi * freq / fs  # [1,1,num_freq]

        # (3) Gain_dB -> A = 10^(Gain_dB/40)
        A = torch.clamp(10.0 ** (Gain_dB / 40.0), min=1e-6)


        # (4) 중심 주파수 Fc -> w0
        epsilon_fc = 1e-3  # 최소 주파수 (Hz)
        Fc = torch.clamp(Fc, min=epsilon_fc)
        epsilon_q = 1e-6
        Q = torch.clamp(Q, min=epsilon_q)

        w0 = 2.0 * math.pi * Fc / fs  # [batch_size, num_filters, 1]
        alpha = torch.sin(w0) / (2.0 * Q)

        cos_w0 = torch.cos(w0)

        # (5) biquad 계수 계산 (배치 크기 유지)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A

        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        # (6) e^{-j w} = cos(w) - j sin(w), e^{-j 2w} = (e^{-j w})^2
        #    w.shape: [1,1,num_freq]
        cos_w = torch.cos(w)
        sin_w = torch.sin(w)
        z1 = torch.complex(cos_w, -sin_w)   # e^{-j w}
        z2 = z1 * z1              # e^{-j 2w}

        b0_c = torch.complex(b0, torch.zeros_like(b0))
        b1_c = torch.complex(b1, torch.zeros_like(b1))
        b2_c = torch.complex(b2, torch.zeros_like(b2))
        a0_c = torch.complex(a0, torch.zeros_like(a0))
        a1_c = torch.complex(a1, torch.zeros_like(a1))
        a2_c = torch.complex(a2, torch.zeros_like(a2))


        # (7) 분자/분모
        # b0,b1,b2,a0,a1,a2: [batch_size, num_filters, 1]
        # z1,z2: [1,1,num_freq]
        # => 브로드캐스트 후 최종 shape: [batch_size, num_filters, num_freq]
        num = b0_c + b1_c * z1 + b2_c * z2
        den = a0_c + a1_c * z1 + a2_c * z2
        H = num / den  # 복소 주파수 응답


        # (8) 진폭(dB), 위상(도) 계산
        #  - 진폭 = 20 log10(|H|)
        #  - 위상 = angle(H) [rad] -> deg
        H_abs = torch.abs(H)
        H_phase = torch.angle(H)
        H_amp_dB = 20.0 * torch.log10(H_abs + 1e-15)
        H_phase_deg = H_phase * 180.0 / math.pi


        return H_amp_dB, H_phase_deg

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) d1/d2/d3 각각 여러 필터(직렬 연결) → dB, 위상 각각 합산
    # ─────────────────────────────────────────────────────────────────────────────

    # -- d1
    Fc_d1   = d1_params[:, :, 0].unsqueeze(-1)  # [batch_size, 5, 1]
    Gain_d1 = d1_params[:, :, 1].unsqueeze(-1)
    Q_d1    = d1_params[:, :, 2].unsqueeze(-1)
    H_d1_dB, H_d1_deg = compute_parametric_peak_response(Fc_d1, Gain_d1, Q_d1, freq, fs=fs)
    # 직렬 연결 → 진폭(dB)는 합, 위상(도)도 합
    H_d1_amp_sum = H_d1_dB.sum(dim=1)   # [batch_size, num_freq]
    H_d1_phase_sum = H_d1_deg.sum(dim=1)

    # -- d2
    Fc_d2   = d2_params[:, :, 0].unsqueeze(-1)
    Gain_d2 = d2_params[:, :, 1].unsqueeze(-1)
    Q_d2    = d2_params[:, :, 2].unsqueeze(-1)
    H_d2_dB, H_d2_deg = compute_parametric_peak_response(Fc_d2, Gain_d2, Q_d2, freq, fs=fs)
    H_d2_amp_sum = H_d2_dB.sum(dim=1)
    H_d2_phase_sum = H_d2_deg.sum(dim=1)

    # -- d3
    Fc_d3   = d3_params[:, :, 0].unsqueeze(-1)
    Gain_d3 = d3_params[:, :, 1].unsqueeze(-1)
    Q_d3    = d3_params[:, :, 2].unsqueeze(-1)
    H_d3_dB, H_d3_deg = compute_parametric_peak_response(Fc_d3, Gain_d3, Q_d3, freq, fs=fs)
    H_d3_amp_sum = H_d3_dB.sum(dim=1)
    H_d3_phase_sum = H_d3_deg.sum(dim=1)

    # -- d4
    Fc_d4   = d4_params[:, :, 0].unsqueeze(-1)
    Gain_d4 = d4_params[:, :, 1].unsqueeze(-1)
    Q_d4    = d4_params[:, :, 2].unsqueeze(-1)
    H_d4_dB, H_d4_deg = compute_parametric_peak_response(Fc_d4, Gain_d4, Q_d4, freq, fs=fs)
    H_d4_amp_sum = H_d4_dB.sum(dim=1)
    H_d4_phase_sum = H_d4_deg.sum(dim=1)

    # -- batch_size, num_drivers, num_freq
    #    드라이버 순서: d1, d2, d3, d4
    H_amp_stack = torch.stack([H_d1_amp_sum, H_d2_amp_sum, H_d3_amp_sum, H_d4_amp_sum], dim=1)
    H_phase_stack = torch.stack([H_d1_phase_sum, H_d2_phase_sum, H_d3_phase_sum, H_d4_phase_sum], dim=1)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) 원본 SPL(dB), Phase(deg)에 결과를 적용
    # ─────────────────────────────────────────────────────────────────────────────
    # SPL : 원본 dB + 필터 dB (직렬 연결 → 합)
    # Phase: 원본 deg + 필터 deg (직렬 연결 → 합)

    # - 브로드캐스트 위해 [batch_size, num_drivers, 1, num_freq]
    H_amp_stack = H_amp_stack.unsqueeze(2)   # [batch_size, num_drivers, 1, num_freq]
    H_phase_stack = H_phase_stack.unsqueeze(2)

    spl_filtered = spl_orig + H_amp_stack
    phase_filtered = phase_orig + H_phase_stack

    # 위상 범위 -180~180 내로 정규화(optional)
    #   (phase + 180) % 360 - 180
    #   pytorch의 경우 float 텐서에 대해 % 연산이 그대로 되므로 아래와 같이 가능
    phase_filtered = (phase_filtered + 180.0) % 360.0 - 180.0

    # ─────────────────────────────────────────────────────────────────────────────
    # 5) 최종 텐서 조합
    # ─────────────────────────────────────────────────────────────────────────────
    # 마지막 차원: [freq(Hz), spl(dB), phase(deg)]
    batched_data_filtered = torch.stack([
        batched_data[..., 0],      # freq 그대로
        spl_filtered,              # 새 SPL
        phase_filtered             # 새 Phase
    ], dim=-1)

    return batched_data_filtered

def crossover_p(batched_data: torch.Tensor, batch_params: torch.Tensor):
    device = batched_data.device
    B, D, A, F, _ = batched_data.shape
    if D != 4:
        raise ValueError("이 예제는 driver 4개인 경우만 가정합니다.")
    
    # 1) freq 축 가져오기 (모든 배치, 드라이버, 앵글 동일 가정)
    freq = batched_data[0, 0, 0, :, 0].to(device)  # shape=[F]
    
    # 2) SPL, Phase 원본
    spl_orig = batched_data[..., 1]   # [B, D, A, F]
    phase_orig = batched_data[..., 2] # [B, D, A, F]
    
    # 3) 차단주파수
    f_c4 = batch_params[:, 48].view(B, 1).to(device)
    f_c1 = batch_params[:, 49].view(B, 1).to(device)
    f_c2 = batch_params[:, 50].view(B, 1).to(device)
    
    amp_d4, phase_d4 = linkwitz_riley_analog(freq, f_c4, filter_type='hpf', slope=24)
    amp_d1, phase_d1 = linkwitz_riley_analog(freq, f_c1, filter_type='lpf', slope=24)
    amp_d2, phase_d2 = linkwitz_riley_analog(freq, f_c2, filter_type='hpf', slope=24)
    amp_d3 = torch.zeros_like(amp_d1)
    phase_d3 = torch.zeros_like(phase_d1)
    
    # angle 차원(A)을 위한 unsqueeze => shape [B, 1, F]
    amp_d1 = amp_d1.unsqueeze(1)   # [B, 1, F]
    amp_d2 = amp_d2.unsqueeze(1)
    amp_d3 = amp_d3.unsqueeze(1)
    amp_d4 = amp_d4.unsqueeze(1)
    ph_d1  = phase_d1.unsqueeze(1)
    ph_d2  = phase_d2.unsqueeze(1)
    ph_d3  = phase_d3.unsqueeze(1)
    ph_d4  = phase_d4.unsqueeze(1)
    
    # 최종 덧셈 (dB/deg)
    # 드라이버 순서는 (d1, d2, d3, d4)
    driver_amp_stack = torch.stack([amp_d1, amp_d2, amp_d3, amp_d4], dim=1)  # [B, 4, 1, F]
    driver_phase_stack = torch.stack([ph_d1, ph_d2, ph_d3, ph_d4], dim=1)    # [B, 4, 1, F]
    
    # SPL = 원본(dB) + 필터(dB), Phase = 원본(deg) + 필터(deg)
    spl_filtered = spl_orig + driver_amp_stack  # [B, 4, A, F]
    phase_filtered = phase_orig + driver_phase_stack
    
    # 위상 래핑 -180 ~ +180
    phase_filtered = (phase_filtered + 180.0) % 360.0 - 180.0
    
    # freq는 기존과 동일
    out = torch.stack([
        batched_data[..., 0],  # freq
        spl_filtered,
        phase_filtered
    ], dim=-1)  # [B, 4, A, F, 3]
 
    return out

def plot_normalized_spl_tensor(tensor_data, title):
    """
    새로운 텐서 자료구조 [batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)]에 
    대해 정규화된 SPL을 시각화하는 함수.

    Parameters:
    - tensor_data: numpy.ndarray 또는 torch.Tensor
        형태가 [batch_size, num_drivers, num_angles, num_freq, 3]인 텐서 데이터.
        여기서 3은 (frequency, SPL, phase)를 의미.
    - title: str
        그래프의 제목.
    """
    # 텐서가 PyTorch 텐서인지 확인하고, 그렇다면 CPU로 이동 후 NumPy 배열로 변환
    if isinstance(tensor_data, torch.Tensor):
        tensor_data = tensor_data.detach().cpu().numpy()

    # 데이터의 형태 확인
    if tensor_data.ndim != 5 or tensor_data.shape[-1] != 3:
        raise ValueError("tensor_data의 형태는 [batch_size, num_drivers, num_angles, num_freq, 3] 이어야 합니다.")
    
    # batch_size=1, num_drivers=1을 가정하고 데이터 추출
    data = tensor_data[0, 0]  # shape: [num_angles, num_freq, 3]
    
    # 주파수와 SPL 데이터 추출
    frequencies = data[0, :, 0]  # 모든 각도에 대해 동일한 주파수 가정
    spl = data[:, :, 1]  # 각도별 SPL 데이터
    
    # 새로운 각도 범위 설정 (-180도부터 180도까지, 10도 간격)
    new_angles = np.arange(-180, 181, UNIT_DEGREE)
    new_data = np.zeros((len(new_angles), len(frequencies)))
    
    # 기존 데이터가 0도부터 시작한다고 가정하고, 음수 각도에 대해서는 절댓값 인덱스 사용
    for i, angle in enumerate(new_angles):
        if angle >= 0:
            index = angle // UNIT_DEGREE
        else:
            index = abs(angle) // UNIT_DEGREE
        if index < spl.shape[0]:
            new_data[i] = spl[index]
        else:
            # 인덱스가 범위를 벗어나는 경우 0으로 채움
            new_data[i] = 0
    
    # 시각화를 위한 meshgrid 생성
    angle_grid = np.linspace(new_angles.min(), new_angles.max(), 500)
    frequency_grid = np.logspace(np.log10(50), np.log10(20000), 500)
    X, Y = np.meshgrid(frequency_grid, angle_grid)
    
    # 데이터 보간
    points = np.array([(f, a) for a in new_angles for f in frequencies])
    values = new_data.flatten()
    Z = griddata(points, values, (X, Y), method='linear')
    
    # 색상 맵 정의
    cmap_colors = [
        '#800000', '#C60000', '#FF1C00', '#FFAA00', '#C6FF39', 
        '#39FFC6', '#00AAFF', '#001CFF', '#0000C6', '#000080'
    ]
    cmap = ListedColormap(cmap_colors[::-1])
    
    # 경계와 정규화 설정
    boundaries = np.linspace(-30, 0, 11)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # 그림 크기 조정 (높이를 줄임)
    fig, ax = plt.subplots(figsize=(12, 8 * 0.7452))
    
    # 컬러 맵을 사용한 pcolormesh 생성
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, norm=norm)
    
    # 축 설정
    ax.set_xscale('log')
    ax.set_xlim([50, 20000])
    ax.set_ylim([-180, 180])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_yticks(np.arange(-180, 181, 30))
    
    # 컬러바 추가
    cbar = fig.colorbar(c, ax=ax, label='Normalized SPL (dB)')
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f'{int(x)}' for x in boundaries])
    
    # 제목 설정
    ax.set_title(title)
    
    # 레이아웃 조정 및 출력
    plt.tight_layout()
    plt.show()

def main():
    # 1) 데이터 로드 (CPU 상에서 로드 후 to(device))
    data_4d = load_data()  # shape [4, num_angles, num_freq, 3]
    data_4d_ver = load_data_ver()

    tensor = torch.zeros((1, 51))
    values = [
        2083.0, -3.3, 1.37,
        2676.0, -1.2, 5.06,
        6677.0, -1.6, 1.3,
        9910.0, 2.2, 2.76,
        14841.0, -1.4, 3,
        15000.0, 8.0, 1,
        17600.0, 0.0, 1.0,
        17500.0, 0.0, 1.0, # d4
        858.0, -2.3, 2.16,
        2150.0, -1.1, 5.64,
        3900.0, 0.0, 1.0,
        4000.0, 0.0, 1.0, # d1
        2200.0, -20.0, 1.0,
        1000.0, -10.0, 1.0, # d2
        2200.0, -20.0, 1.0,
        1000.0, -10.0, 1.0, # d3
        2000.0, 2000.0, 20.0,
    ]
    tensor[0, :len(values)] = torch.tensor(values)
    data_4d = peak_p(data_4d.unsqueeze(0),tensor)
    data_4d = crossover_p(data_4d,tensor)
    data_4d[0, 1, :, :, 2] = (data_4d[0, 1, :, :, 2] + 360.0) % 360.0 - 180.0
    data_4d[0, 0, :, :, 1] = data_4d[0, 0, :, :, 1] - 4
    data_4d[0, 2, :, :, 1] = data_4d[0, 0, :, :, 1] - 4
    data_4d[0, 3, :, :, 1] = data_4d[0, 3, :, :, 1] - 4
    data_4d = data_4d.squeeze(0)

    data_4d_ver = peak_p(data_4d_ver.unsqueeze(0),tensor)
    data_4d_ver = crossover_p(data_4d_ver,tensor)
    data_4d_ver[0, 1, :, :, 2] = (data_4d_ver[0, 1, :, :, 2] + 360.0) % 360.0 - 180.0
    data_4d_ver[0, 0, :, :, 1] = data_4d_ver[0, 0, :, :, 1] - 4
    data_4d_ver[0, 2, :, :, 1] = data_4d_ver[0, 0, :, :, 1] - 4
    data_4d_ver[0, 3, :, :, 1] = data_4d_ver[0, 3, :, :, 1] - 4
    data_4d_ver = data_4d_ver.squeeze(0)
    
    # 2) PSO 인스턴스 생성
    pso = ParticleSwarmOptimizer(
        data=data_4d,
        num_particles=NUM_PARTICLES,
        max_iter=NUM_ITERATIONS,
        w=0.6,
        c1=1.5,
        c2=1.5,
        device=device  # ex) torch.device('cuda')
    )
    
    # log.txt 파일을 열고, Tee를 통해 stdout과 log 파일에 동시에 출력되도록 설정
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    # 3) PSO 실행
    best_params, best_cost = pso.optimize(objective, data_4d_ver)

    # 결과 확인
    print("=== PSO 최적화 결과 ===")
    print(f"Best Cost : {best_cost:.4f}")

    print("\nD1 필터 파라미터 (5개):")
    for i in range(5):
        fc   = best_params[i*3 + 0]
        gain = best_params[i*3 + 1]
        qval = best_params[i*3 + 2]
        print(f"  Filter {i+1}: Freq = {fc:.2f} Hz, Gain = {gain:.2f} dB, Q = {qval:.2f}")
    print(f"D1 HPF crossover freq: {best_params[75]:.2f} Hz")

    print("\nD2 필터 파라미터 (10개):")
    for i in range(10):
        fc   = best_params[15 + i*3 + 0]
        gain = best_params[15 + i*3 + 1]
        qval = best_params[15 + i*3 + 2]
        print(f"  Filter {i+1}: Freq = {fc:.2f} Hz, Gain = {gain:.2f} dB, Q = {qval:.2f}")
    print(f"D2 LPF crossover freq: {best_params[76]:.2f} Hz")
    print(f"D2 Delay: {best_params[78]:.2f} us")

    print("\nD3 필터 파라미터 (10개):")
    for i in range(10):
        fc   = best_params[45 + i*3 + 0]
        gain = best_params[45 + i*3 + 1]
        qval = best_params[45 + i*3 + 2]
        print(f"  Filter {i+1}: Freq = {fc:.2f} Hz, Gain = {gain:.2f} dB, Q = {qval:.2f}")
    print(f"D3 LPF crossover freq: {best_params[77]:.2f} Hz")
    print(f"D3 Delay: {best_params[79]:.2f} us")


    # 로그 파일을 닫아 저장 완료
    log_file.close()

if __name__ == "__main__":
    main()



"""
data: 4d tensor 		[num_drivers, num_angles, num_freq, (freq, spl, phase)]
batching: 5d tensor		[batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)]

params: 1d tensor		[num_params=80]	-> 1:75 peak, 76:78 crossover, 79:80 delay
batching: 2d tensor		[batch_size, num_params=80]

consider phase invert @ load_data when pilepath = 'd2'

load_data(filepath)						    return 4d tensor [num_drivers, num_angles, num_freq, (freq, spl, phase)]
peak(batched data, batched params)			return 5d tensor [batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)]
crossover(batched data, batched params)		return 5d tensor [batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)]
delay(batched data, batched params)			return 5d tensor [batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)]
combine_and_normalize(batched data)			return 5d tensor [batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)]
objective(batched data, batched params)     return 2d tensor [batch_size, 1]    

main
1. load_data:		    output [num_drivers, num_angles, num_freq, (freq, spl, phase)]
2. call pso		        input data and params. each [num_drivers, num_angles, num_freq, (freq, spl, phase)] and [num_params=80]
3. batching		        -> [batch_size, num_drivers, num_angles, num_freq, (freq, spl, phase)] and -> [batch_size, num_params=80]. call objective function
4. objective	    	call peak, crossover, delay with the batched parameters. update best params with cost.
5. result		        print best params when pso finished
"""


