import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
#
def main():
    parser = argparse.ArgumentParser(description='Koopa用于时间序列预测')

    # 基本配置
    parser.add_argument('--is_training', type=int, default=1, help='训练状态')
    parser.add_argument('--model_id', type=str, default='test', help='模型ID')
    parser.add_argument('--model', type=str, default='Koopa', help='模型名称，选项：[Koopa]')

    # 数据加载器
    parser.add_argument('--data', type=str, default='custom', help='数据集类型')
    #custom 为默认参数
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='数据文件的根路径')

    #换数据
    # default = './dataset/ETT-small/ .表示数据在主文件夹（run）
    #输出列的表头是OT，输入列的表头只能是英文或数字不能有中文。时间列在第一列，表头是date
    parser.add_argument('--data_path', type=str, default='YH5date.csv', help='数据文件')
    # 具体数据
    parser.add_argument('--features', type=str, default='MS', help='预测任务，选项：[M, S, MS]; M:多变量预测多变量，S:单变量预测单变量，MS:多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT', help='S或MS任务中的目标特征')
    parser.add_argument('--freq', type=str, default='d', help='时间特征编码的频率，选项：[s:每秒，t:每分钟，h:每小时，d:每日，b:工作日，w:每周，m:每月]，也可以使用更详细的频率如15min或3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的位置')

    # 预测任务
    parser.add_argument('--seq_len', type=int, default=4, help='输入序列长度')#96
    parser.add_argument('--label_len', type=int, default=2, help='开始标记长度')#48
    parser.add_argument('--pred_len', type=int, default=4, help='预测序列长度')#48

    # 模型定义
    parser.add_argument('--enc_in', type=int, default=8 ,help='编码器输入大小')
    parser.add_argument('--dec_in', type=int, default=8, help='解码器输入大小')
    parser.add_argument('--c_out', type=int, default=8, help='输出大小')
    #输入的特征列数+1（除去时间和预测列）
    parser.add_argument('--dropout', type=float, default=0.3, help='丢弃率')  # 0.05
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码，选项：[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true', help='是否预测未见过的未来数据')

    # 优化
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载器的工作进程数')
    parser.add_argument('--itr', type=int, default=1, help='实验次数')
    #试验次数10-500
    parser.add_argument('--train_epochs', type=int, default=40, help='训练轮数')
    # default=为默认参数（10-500）
    parser.add_argument('--batch_size', type=int, default=64, help='训练输入数据的批量大小')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='优化器学习率')#0.001
    # 优化学习率调整范围0.1-0.0001
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    parser.add_argument('--loss', type=str, default='mse', help='损失函数')#mse
    parser.add_argument('--lradj', type=str, default='type1', help='调整学习率')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多个GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU的设备ID')
    parser.add_argument('--seed', type=int, default=2023, help='随机种子')

    # Koopa
    parser.add_argument('--dynamic_dim', type=int, default=128, help='Koopman嵌入的潜在维度')#128
    parser.add_argument('--hidden_dim', type=int, default=64, help='en/decoder的隐藏维度')#64
    parser.add_argument('--hidden_layers', type=int, default=3, help='en/decoder的隐藏层数')
    parser.add_argument('--seg_len', type=int, default=1, help='时间序列的分段长度')#48
    parser.add_argument('--num_blocks', type=int, default=3, help='Koopa块的数量')#3
    parser.add_argument('--alpha', type=float, default=0.2, help='频谱滤波器比率')
    parser.add_argument('--multistep', action='store_true', help='是否对多步K使用近似', default=False)
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数量')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    print('实验参数:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # 实验设置记录
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_segl{}_dyna{}_h{}_l{}_nb{}_a{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.seg_len,
                args.dynamic_dim,
                args.hidden_dim,
                args.hidden_layers,
                args.num_blocks,
                args.alpha,
                args.des, ii)

            exp = Exp(args)  # 设置实验
            print('>>>>>>>开始训练: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>测试: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>预测: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()#清理当前未使用的缓存内存
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_segl{}_dyna{}_h{}_l{}_nb{}_a{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.seg_len,
            args.dynamic_dim,
            args.hidden_dim,
            args.hidden_layers,
            args.num_blocks,
            args.alpha,
            args.des, ii)

        exp = Exp(args)  # 设置实验
        print('>>>>>>>测试: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


