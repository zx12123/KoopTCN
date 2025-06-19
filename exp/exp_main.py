import pandas as pds
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import os
import time
import warnings
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Koopa
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from sklearn.preprocessing import StandardScaler
from data_provider.data_loader import Dataset_Custom


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.scaler = None  # 初始化 self.scaler
        self.__read_data__(args)  # 传递 args 以供数据读取
    def __read_data__(self, args):
        # 创建 Dataset_Custom 实例
        dataset = Dataset_Custom(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='train',  # 可以根据需要调整为 'train', 'val' 或 'test'
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=True,  # 如果需要标准化数据
            timeenc=0,  # 根据需要
            freq='d'  # 频率可以根据需要调整
        )
        self.scaler = dataset.scaler  # 获取 scaler
        return dataset


    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_data, train_loader = self._get_data(flag='train')
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0]*self.args.alpha)).indices
        return mask_spectrum # as the spectrums of time-invariant component




    def _build_model(self):
        model_dict = {
            'Koopa': Koopa,
        }
        self.args.mask_spectrum = self._get_mask_spectrum()
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_raw) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):
        global scaler
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        output_folder = './fourier_results/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # zx创建文件夹

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_raw) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        preds = []
        trues = []
        batch_ys = []  # 用于存储所有批次的 batch_x 值



        #创建一个文件夹来保存测试结果，确保文件夹路径存在。
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,seq_y_raw) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y1 = batch_y[:, -1:, :].to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]    #取出最后 pred_len 个时间步的数据
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 从标签中获取最后 pred_len 个时间步的数据。


                pred =outputs #.detach().cpu().numpy()#outputs.squeeze()
                true =batch_y #.detach().cpu().numpy()#batch_y.squeeze()

                preds.append(pred)
                trues.append(true)
                batch_ys.append(batch_y1)  # 存储当前批次的 batch_x


                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('preds.shape:' , preds.shape)
        print('trues.shape:' , trues.shape)
        batch_ys= np.array(batch_ys)

        # ZX：标准化真实数据集和预测数据集形状由（1043，48，1）转换为 (1043, 48)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('preds.shape:', preds.shape)
        print('trues.shape:', trues.shape)

        batch_ys_rescaled = self.scaler.inverse_transform(batch_ys.reshape(-1, batch_ys.shape[-1]))
        batch_ys_rescaled = batch_ys_rescaled.reshape(batch_ys.shape)
        print(batch_ys.shape)
        # 将 batch_x 输出到 Excel
        # 将反标准化后的 batch_x 输出到 Excel
        df_batch_y_rescaled = pds.DataFrame(batch_ys_rescaled.reshape(batch_ys_rescaled.shape[0], -1))  # 转换为 DataFrame
        df_batch_y_rescaled.to_excel(os.path.join(folder_path, "trues_rescaled_values.xlsx"),
                                     index=False)
        print('trues values have been saved to Excel.')

        # 将 batch_ys 从 (1043, 1, 1, 8) 转换为 (1043, 8)
        batch_ys_reshaped = batch_ys.reshape(-1, batch_ys.shape[-1])  # 转换为 (1043, 8)

        # 用 preds 的最后一个时间步数据替换 batch_ys 的最后一列
        last_step_pred = preds[:, -1, :]  # 获取 preds 最后时间步的数据
        batch_ys_reshaped[:, -1] = last_step_pred[:, -1] # 替换最后一列

        # 对新的数组进行反标准化
        batch_ys_rescaled_final = self.scaler.inverse_transform(batch_ys_reshaped)

        # 保存处理后的反标准化结果到 Excel
        df_batch_ys_rescaled_final = pds.DataFrame(batch_ys_rescaled_final)  # 转换为 DataFrame
        df_batch_ys_rescaled_final.to_excel(os.path.join(folder_path, "batch_ys_rescaled_final.xlsx"),
                                            index=False)  # 保存到 Excel
        print('preds values have been saved to Excel.')

        # 将预测值和真实值输出到 Excel
        df_preds = pds.DataFrame(preds.reshape(preds.shape[0], -1))  # 转换为 DataFrame
        df_trues = pds.DataFrame(trues.reshape(trues.shape[0], -1))  # 转换为 DataFrame

        # 保存预测值和真实值到 Excel
        df_preds.to_excel(os.path.join(folder_path, "predictions.xlsx"), index=False)  # 保存到 Excel
        df_trues.to_excel(os.path.join(folder_path, "true_values.xlsx"), index=False)  # 保存到 Excel

        print('Predictions and true values have been saved to Excel.')

        print('test_data:', preds.shape, trues.shape)


        #result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        # 保存指标和预测结果
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        actuals = []  # 用于存储实际值
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_raw) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                actuals.append(batch_y.detach().cpu().numpy())  # 实际值

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        actuals = actuals.reshape(-1, actuals.shape[-2], actuals.shape[-1])  # 重新调整实际值的形状

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        np.save(folder_path + 'real_prediction.npy', preds)

        return

