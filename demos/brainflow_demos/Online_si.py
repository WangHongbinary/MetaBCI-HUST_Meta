# -*- coding: utf-8 -*-
# License: MIT License
"""
SI Feedback on Neuracle.
"""

import time
import numpy as np
import os
import torch
import mne

from torch import nn
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import Marker, Neuracle_DSI, Neuracle
from metabci.brainflow.workers import ProcessWorker
from metabci.brainflow.logger import get_logger
from mne.io import read_raw_fif
from metabci.brainda.algorithms.deep_learning import EEGNet
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import fractional_matrix_power


# basic experiment control
import serial  
import numpy as np
from time import sleep

logger_online = get_logger("si_online")

ser = serial.Serial()

def port_open_recv():  # 对串口的参数进行配置
    ser.port = 'COM4' 
    ser.baudrate = 115200
    ser.bytesize = 8
    ser.stopbits = 1
    ser.parity = "N"  # 奇偶校验位
    ser.open()
    ser.flushOutput()
    if ser.isOpen():
        print("串口打开成功！")
    else:
        print("串口打开失败！")


class CarMove():
    def __init__(self):
        pass
    
    def BYTE0(self, x):
        return x & 0xFF          # 取最低有效字节

    def BYTE1(self, x):
        return (x >> 8) & 0xFF   # 取次低有效字节

    def BYTE2(self, x):
        return (x >> 16) & 0xFF  # 取次高有效字节

    def BYTE3(self, x):
        return (x >> 24) & 0xFF  # 取最高有效字节

    def port_close(self):
        ser.close()
        if ser.isOpen():
            print("串口关闭失败！")
        else:
            print("串口关闭成功！")

    def send(self, send_data):
        if ser.isOpen():
            try:
                data_bytes = bytes(send_data) # 将输入的数组转换为字节
                ser.write(data_bytes)  # 发送字节数据
                print("发送成功", send_data)
            except Exception as e:
                print("发送失败！", str(e))
        else:
            print("发送失败！")

    def move_pos(self, x, y, spd):
        # 定义要发送的数据数组
        x = np.int32(x)
        y = np.int32(y)
        z = np.int32(0)
        spd = np.int16(spd)
        sum = np.uint16(0)
        data_to_send = [0xDF, 0x01, 0x97, 0x02, 0x65, 14, self.BYTE0(x), self.BYTE1(x), self.BYTE2(x), self.BYTE3(x), 
                        self.BYTE0(y), self.BYTE1(y), self.BYTE2(y), self.BYTE3(y), self.BYTE0(z), self.BYTE1(z), self.BYTE2(z), self.BYTE3(z),
                        self.BYTE0(spd), self.BYTE1(spd), 0xFD] 
        for i in range(data_to_send[5]+7):
            sum += data_to_send[i] 
        data_to_send = [0xDF, 0x01, 0x97, 0x02, 0x65, 14, self.BYTE0(x), self.BYTE1(x), self.BYTE2(x), self.BYTE3(x), 
                        self.BYTE0(y), self.BYTE1(y), self.BYTE2(y), self.BYTE3(y), self.BYTE0(z), self.BYTE1(z), self.BYTE2(z), self.BYTE3(z),
                        self.BYTE0(spd), self.BYTE1(spd), 0xFD, self.BYTE0(sum), self.BYTE1(sum)]
        self.send(data_to_send)  # 发送数组

    def move_rot(self, z, spd):
        # 定义要发送的数据数组
        x = np.int16(0)
        y = np.int16(0)
        z = np.int32(z)
        spd = np.int16(spd)
        sum = np.uint16(0)
        data_to_send = [0xDF, 0x01, 0x97, 0x02, 0x66, 0x0A, self.BYTE0(x), self.BYTE1(x), self.BYTE0(y), self.BYTE1(y), 
                        self.BYTE0(z), self.BYTE1(z), self.BYTE2(z), self.BYTE3(z), self.BYTE0(spd), self.BYTE1(spd), 0xFD]
        for i in range(data_to_send[5]+7):
            sum += data_to_send[i]
        data_to_send = [0xDF, 0x01, 0x97, 0x02, 0x66, 0x0A, self.BYTE0(x), self.BYTE1(x), self.BYTE0(y), self.BYTE1(y), 
                        self.BYTE0(z), self.BYTE1(z), self.BYTE2(z), self.BYTE3(z), self.BYTE0(spd), self.BYTE1(spd), 0xFD, self.BYTE0(sum), self.BYTE1(sum)]
        self.send(data_to_send)  # 发送数组

    def move_yawoffset(self, offset):
        offset = np.int32(offset)
        sum = np.uint16(0)
        data_to_send = [0xDF, 0x01, 0x97, 0x09, 0x6F, 0x06, self.BYTE0(offset), self.BYTE1(offset), self.BYTE2(offset), self.BYTE3(offset),
                        self.BYTE0(1), self.BYTE0(1), 0xFD]
        for i in range(data_to_send[5]+7):
            sum += data_to_send[i]
        data_to_send = [0xDF, 0x01, 0x97, 0x09, 0x6F, 0x06, self.BYTE0(offset), self.BYTE1(offset), self.BYTE2(offset), self.BYTE3(offset),
                        self.BYTE0(1), self.BYTE0(1), 0xFD, self.BYTE0(sum), self.BYTE1(sum)]
        self.send(data_to_send)  # 发送数组


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


def EA(x):
    new_x = np.zeros_like(x) # (N,C,T)
    cov = np.zeros((x.shape[0],x.shape[1],x.shape[1])) 
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov,0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    new_x = np.matmul(sqrtRefEA, x)
    return new_x


def read_data(filepath, pick_chs, stim_interval, stim_labels):
    
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('n')[1].split('_')[0]))

    all_epochs = []

    for file in file_list:
        file_name = os.path.join(filepath, file)
        raw = read_raw_fif(file_name, preload=True)
        events = mne.find_events(raw, stim_channel='Trigger')
        print(events)
        
        raw.filter(l_freq=70, h_freq=140)
        raw.pick_channels(pick_chs)
        raw.set_eeg_reference('average')

        epochs = mne.Epochs(raw, 
                            events, 
                            event_id=stim_labels, 
                            tmin=stim_interval[0], 
                            tmax=stim_interval[1], 
                            baseline=(-0.5, 0), 
                            detrend=1, 
                            preload=True)
        
        all_epochs.append(epochs)
    
    combined_epochs = mne.concatenate_epochs(all_epochs)

    Xs = combined_epochs.get_data()[..., :2100]
    ys = combined_epochs.events[:, 2]
    print('ys.shape', ys.shape)
    print('ys', ys)

    ys = label_encoder(ys, stim_labels)

    print('Xs.shape', Xs.shape)
    print('ys.shape', ys.shape)
    print('ys', ys)

    return Xs, ys


# 训练模型


def train_model(tr_X, va_X, tr_y, va_y, model_path, max_epoch, device):

    tr_set = EEGDataset(tr_X, tr_y)
    va_set = EEGDataset(va_X, va_y)

    tr_loader = DataLoader(tr_set, batch_size = 32, shuffle = True, drop_last = False)
    va_loader = DataLoader(va_set, batch_size = 40, shuffle = True, drop_last = False)
 
    model = EEGNet(n_channels=tr_X.shape[-2], n_samples=tr_X.shape[-1], n_classes=len(np.unique(tr_y))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_f = nn.CrossEntropyLoss()

    best_valid_loss = np.inf
    best_epoch = -1

    for epoch in range(max_epoch):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in tr_loader:
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            batch_loss = loss_f(output, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item() * X.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += X.size(0)
            train_correct += (predicted == y).sum().item()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        for X, y in va_loader:
            X = X.to(device)
            y = y.to(device)

            with torch.no_grad():
                output = model(X)
            batch_loss = loss_f(output, y)

            valid_loss += batch_loss.item() * X.size(0)
            _, predicted = torch.max(output.data, 1)
            valid_total += X.size(0)
            valid_correct += (predicted == y).sum().item()

        valid_acc = valid_correct / valid_total
        avg_valid_loss = valid_loss / valid_total

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), os.path.join(model_path, 'Net.pth'))

        print(f"Epoch {epoch+1}/{max_epoch} | Train Acc: {train_acc:.4f} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {valid_acc:.4f} | Val Loss: {avg_valid_loss:.4f}")
    print(f"\nBest validation model from epoch {best_epoch+1}")


# 预测标签


def model_predict(X, model=None, device=None):
    X = X.to(device)

    with torch.no_grad():
        output = model(X)

    print(output)

    _, p_labels = torch.max(output.data, 1)

    return p_labels


# 计算离线正确率


def offline_validation(X, y, model_path, is_EA, device):
    max_epoch = 10
    print('X.shape:', X.shape)

    X_train, y_train = X[0:448], y[0:448]
    X_valid, y_valid = X[448:488], y[448:488]

    if is_EA == True:
        X_train = EA(X_train)
        X_valid = EA(X_valid)

    X_train = torch.tensor(X_train, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.long)
    X_valid = torch.tensor(X_valid, dtype = torch.float32)
    y_valid = torch.tensor(y_valid, dtype = torch.long)

    train_model(X_train, 
                X_valid, 
                y_train, 
                y_valid, 
                model_path=model_path, 
                max_epoch=max_epoch, 
                device=device)
    
    model = EEGNet(n_channels=X_train.shape[-2], n_samples=X_train.shape[-1], n_classes=len(np.unique(y_train))).to(device)
    state_dict = torch.load(os.path.join(model_path, 'Net.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    
    p_labels = model_predict(X_valid, 
                             model=model, 
                             device=device)

    valid_acc = (p_labels == y_valid).sum().item() / len(y_valid)
    return np.array(valid_acc)


class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 filepath,
                 pick_chs,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 offline_train,
                 is_EA,
                 is_car,
                 timeout,
                 worker_name):
        self.filepath = filepath
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.offline_train = offline_train
        self.is_EA = is_EA
        self.is_car = is_car

        if self.is_car:
            self.car = CarMove()

        self.model_path = 'F:\\Work_Dell&7900\\BCI2025\\MetaBCI-HUST_Meta\\model_save'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('self.device:', self.device)

        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        if self.offline_train == True:
            # 离线训练模型
            X, y = read_data(self.filepath,
                             self.pick_chs,
                             self.stim_interval,
                             self.stim_labels)
            print("Loding data successfully")

            acc = offline_validation(X, 
                                     y,
                                     model_path=self.model_path, 
                                     is_EA=self.is_EA,
                                     device=self.device)     
            print("Current model accuracy:", acc)

        model = EEGNet(n_channels=20, n_samples=2100, n_classes=4).to(self.device)
        state_dict = torch.load(os.path.join(self.model_path, 'Net.pth'))
        model.load_state_dict(state_dict)
        model.eval()

        self.estimator = model

        # 构建反馈通路连接刺激界面，反馈结果
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        data = np.asarray(data, dtype=np.float32).T
        data[:24, :] = data[:24, :] * 1e-7
        # print("data[0]:", data[0])
        print("data.shape", data.shape)

        mne.set_log_level('CRITICAL')

        ch_names = ['EEG P3-Pz', 
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
                    'Trigger']
        
        ch_types = ['eeg'] * 25
        info = mne.create_info(ch_names=ch_names, sfreq=self.srate, ch_types=ch_types)
        raw_data = mne.io.RawArray(data, info)

        events = mne.find_events(raw_data, stim_channel='Trigger')
        # print('events',events)

        event = events[0]
        stim_labels = event[2]
        # print('stim_labels', stim_labels)

        event = np.expand_dims(event, axis=0)
        # print('event', event)

        # print(raw_data.info)
        # print(self.pick_chs)

        raw_data.filter(l_freq=70, h_freq=140)
        raw_data.pick_channels(self.pick_chs)
        raw_data.set_eeg_reference('average')

        tmin = -2
        tmax = 5

        epoch = mne.Epochs(raw_data, 
                           event, 
                           event_id=stim_labels, 
                           tmin=tmin, 
                           tmax=tmax, 
                           baseline=(-0.5, 0), 
                           detrend=1, 
                           preload=True)

        input_data = epoch.get_data()[..., :2100]

        print('input_data.shape', input_data.shape)
        input_data = torch.tensor(input_data, dtype = torch.float32)

        p_pred = model_predict(input_data, 
                               model=self.estimator, 
                               device=self.device)

        p_pred = int(p_pred)
        p_pred = p_pred + 1
        p_labels = [p_pred]
        print("online p_labels:", p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

        # move the car
        if self.is_car and not ser.isOpen():
            port_open_recv()  # 打开串口
            if p_pred == 1:
                self.car.move_pos(0, 6000, 1000)
            elif p_pred == 2:
                self.car.move_pos(0, -6000, 1000)
            elif p_pred == 3:
                self.car.move_pos(-6000, 0, 1000)
            elif p_pred == 4:
                self.car.move_pos(6000, 0, 1000)
            sleep(3)
        elif self.is_car:
            if p_pred == 1:
                self.car.move_pos(0, 6000, 1000)
            elif p_pred == 2:
                self.car.move_pos(0, -6000, 1000)
            elif p_pred == 3:
                self.car.move_pos(-6000, 0, 1000)
            elif p_pred == 4:
                self.car.move_pos(6000, 0, 1000)
            sleep(3)

    def post(self):
        pass


if __name__ == '__main__':
    srate = 300                         # 放大器的采样率
    stim_interval_offline = [-2, 5]             # 离线截取数据的时间段
    stim_interval_online = [-2.3, 5.3]          # 在线截取数据的时间段
    stim_labels = list(range(1, 5))     # 1:前进 2:后退 3:左转 4:右转

    filepath = "D:\\Speech_MetaBCI\\Subject01\\processed"
    pick_chs = ['EEG P3-Pz', 
                'EEG C3-Pz', 
                'EEG F3-Pz', 
                'EEG Fz-Pz', 
                'EEG F4-Pz', 
                'EEG C4-Pz', 
                'EEG P4-Pz', 
                'EEG Cz-Pz', 
                'EEG A1-Pz', 
                'EEG Fp1-Pz', 
                'EEG Fp2-Pz', 
                'EEG T3-Pz', 
                'EEG T5-Pz', 
                'EEG O1-Pz', 
                'EEG O2-Pz', 
                'EEG F7-Pz', 
                'EEG F8-Pz',  
                'EEG A2-Pz', 
                'EEG T6-Pz', 
                'EEG T4-Pz']

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'
    offline_train = False    # 是否离线训练模型
    is_EA = False
    is_car = False           # 是否接入小车

    worker = FeedbackWorker(filepath=filepath,
                            pick_chs=pick_chs,
                            stim_interval=stim_interval_offline,
                            stim_labels=stim_labels,
                            srate=srate,
                            lsl_source_id=lsl_source_id,
                            offline_train=offline_train,
                            is_EA=is_EA,
                            is_car=is_car,
                            timeout=5e-2,
                            worker_name=feedback_worker_name)  # 在线处理
    marker = Marker(interval=stim_interval_online, srate=srate, events=stim_labels)

    na = Neuracle_DSI(
        device_address=('127.0.0.1', 8844),
        srate=srate,
        num_chans=25)  # NeuroScan parameter
    
    # 与na建立tcp连接
    na.connect_tcp()
    print("Neuracle TCP Connected")
    logger_online.info("Neuracle TCP Connected")

    # na开始采集波形数据
    na.recv()
    print("Neuracle Data Streaming")
    logger_online.info("Neuracle Data Streaming")

    # register worker来实现在线处理
    na.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    na.up_worker(feedback_worker_name)
    # 等待 0.5s
    time.sleep(0.5)

    # ns开始截取数据线程，并把数据传递数据给处理进程
    na.start_trans()

    # 任意键关闭处理进程
    input('press any key to close\n')
    # 关闭处理进程
    na.down_worker('feedback_worker')
    # 等待 1s
    time.sleep(1)

    # na停止在线截取线程
    na.stop_trans()
    
    # na停止采集波形数据
    na.set_timeout()
    # 与na断开连接
    na.close_connection()
    na.clear()
    print('bye')