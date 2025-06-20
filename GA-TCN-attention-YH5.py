from multitcn_components import TCNStack,  DownsampleLayerWithAttention
import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
from bitstring import BitArray
import math
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
import random

np.set_printoptions(suppress=True) #取消科学计数法，易对比观察
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号

### 设定随机种子
print("Enter a seed for the experiment:")
seed = input()
if len(seed)!=0 and seed.isdigit():
    seed = int(seed)
else:
    seed = 192
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



train_ratio = 0.8 #训练集比例1
forecast_horizon = 1  #预测步长
###设定训练参数
epochs = 10
batch_size = 64
#loss = 'mse'
#设定网络参数
tcn_kernel_size = 3
tcn_use_bias = True
tcn_filter_num = 64
tcn_kernel_initializer = 'random_normal'
tcn_dropout_format = "channel"
#tcn_activation = 'leaky_relu'
tcn_final_activation = 'linear'
tcn_final_stack_activation = 'relu'


#读取数据
data_files = []
data_files = pd.read_excel("nanhu8.xlsx")
data_files = np.array(data_files)
data = data_files[:,1:] #去掉时间一列
data = data.astype('float32')  #data(227.13)
feature = data.shape[1]

#分割训练集和测试集
train_size = int(len(data) * train_ratio) #train size = 181
#data_train = data[0:train_size,:] #(181,13)
#data_test = data[train_size:,:] #（46,13）

X_data = data
Y_data = data[:,:1]

#数据归一化处理
preprocessor = preprocessing.MinMaxScaler()
out_preprocessor = preprocessing.MinMaxScaler()


preprocessor.fit(X_data)
data_x_norm = preprocessor.transform(X_data)
preprocessor.fit(Y_data)
data_y_norm = preprocessor.transform(Y_data)

def windowed_dataset(series, time_series_number, window_size):
    """
    Returns a windowed dataset from a Pandas dataframe
    将输入数据进行时间步长处理
    """
    available_examples = series.shape[0]-window_size + 1
    time_series_number = series.shape[1]
    inputs = np.zeros((available_examples,window_size,time_series_number))
    for i in range(available_examples):
        inputs[i,:,:] = series[i:i+window_size,:]
    return inputs

def windowed_forecast(series, forecast_horizon):
    #处理输出数据的时间步长
    available_outputs = series.shape[0]- forecast_horizon + 1
    output_series_num = series.shape[1]
    output = np.zeros((available_outputs,forecast_horizon, output_series_num))
    for i in range(available_outputs):
        output[i,:]= series[i:i+forecast_horizon,:]
    return output

def train_evaluate(ga_individual_solution):
    # 对目标超参数进行解码
    activation_select = ['sigmoid', 'relu', 'leaky_relu', 'tanh']
    loss_select = ['mse', 'mape', 'mae', 'huber_loss']

    learning_rate = BitArray(ga_individual_solution[0:7])
    layer_num = BitArray(ga_individual_solution[7:10])
    dropout_rate = BitArray(ga_individual_solution[10:13])
    windows_length = BitArray(ga_individual_solution[13:18])
    tcn_activation = BitArray(ga_individual_solution[18:20])
    loss = BitArray(ga_individual_solution[20:22])



    dropout_rate = dropout_rate.uint
    learning_rate = learning_rate.uint
    windows_length = windows_length.uint + 1
    layer_num = layer_num.uint + 4
    tcn_activation =tcn_activation.uint
    loss = loss.uint


    learning_rate = learning_rate * (math.exp(-9)) + 0.00001
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    dropout_rate = (dropout_rate + 1) * 0.1
    tcn_activation = activation_select[tcn_activation]
    loss = loss_select[loss]

    print('\n windows_length: ', windows_length, ', layer_num: ', layer_num,
          ', learning_rate: ', learning_rate, ', dropout_rate: ', dropout_rate, 'activation:',tcn_activation ,',loss:', loss)


    #获取训练集测试集真实的y值
    train_y_real = Y_data[windows_length:train_size + windows_length]
    test_y_real = Y_data[train_size + windows_length:]
    #根据获得的窗口长度划分数据集
    data_x = windowed_dataset(data_x_norm[:-forecast_horizon], feature, windows_length)
    data_y = data_y_norm[windows_length:]
    train_x = data_x[0:train_size, :, :]
    train_y = data_y[0:train_size, :]
    test_x = data_x[train_size:, :, :]
    test_y = data_y[train_size:, :]

    #构建网络
    class MTCNAModel(tf.keras.Model):

        def __init__(self, tcn_layer_num, tcn_kernel_size, tcn_filter_num, window_size, forecast_horizon,
                     num_output_time_series, use_bias, kernel_initializer, tcn_dropout_rate, tcn_dropout_format,
                     tcn_activation, tcn_final_activation, tcn_final_stack_activation):
            super(MTCNAModel, self).__init__()

            self.num_output_time_series = num_output_time_series

            # Create stack of TCN layers
            self.lower_tcn = TCNStack(tcn_layer_num, tcn_filter_num, tcn_kernel_size, window_size, use_bias,
                                      kernel_initializer, tcn_dropout_rate, tcn_dropout_format, tcn_activation,
                                      tcn_final_activation, tcn_final_stack_activation)

            self.downsample_att = DownsampleLayerWithAttention(num_output_time_series, window_size, tcn_kernel_size,
                                                               forecast_horizon, kernel_initializer, None)

        def call(self, input_tensor):
            x = self.lower_tcn(input_tensor)
            x, distribution = self.downsample_att([x, input_tensor])
            return x

    model = MTCNAModel(layer_num, tcn_kernel_size, tcn_filter_num, windows_length, forecast_horizon, feature,
                       tcn_use_bias, tcn_kernel_initializer, dropout_rate, tcn_dropout_format, tcn_activation,
                       tcn_final_activation, tcn_final_stack_activation)
    model.compile(optimizer, loss, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    start_time = tf.timestamp()

    history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=0)

    end_time = tf.timestamp()

    duration = end_time - start_time
    test_pre = model.predict(test_x)
    test_pre = np.array(tf.squeeze(test_pre)).reshape(-1, 1)
    preprocessor.fit(Y_data)
    test_pre_real = preprocessor.inverse_transform(test_pre)
    RMSE = np.sqrt(mean_squared_error(test_y_real, test_pre_real))
    print('Validation RMSE: ', RMSE, '\n')
    return RMSE,

population_size = 2 #初始种群数
num_generations = 4 #迭代次数
gene_length = 22  #个体基因长度

#Implementation of Genetic Algorithm using DEAP python library.

#Since we try to minimise the loss values, we use the negation of the root mean squared loss as fitness function.
creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

#initialize the variables as bernoilli random variables
toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

#Ordered cross-over used for mating
toolbox.register('mate', tools.cxOrdered)
#Shuffle mutation to reorder the chromosomes
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
#use roulette wheel selection algorithm 锦标赛
toolbox.register('select', tools.selTournament, tournsize = 2)
#training function used for evaluating fitness of individual solution.
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)

optimal_individuals_data = tools.selBest(population,k = 1) #select top 1 solution
windows_length = None
layer_num = None
learning_rate = None
dropout_rate = None

activation_select = ['relu', 'sigmoid', 'leaky_relu', 'tanh']
loss_select = ['mse', 'mape', 'mae', 'huber_loss']


for bi in optimal_individuals_data:
    learning_rate = BitArray(bi[0:7])
    layer_num = BitArray(bi[7:10])
    dropout_rate = BitArray(bi[10:13])
    windows_length = BitArray(bi[13:18])
    tcn_activation = BitArray(bi[18:20])
    loss = BitArray(bi[20:22])

    dropout_rate = dropout_rate.uint
    learning_rate = learning_rate.uint
    windows_length = windows_length.uint + 1
    layer_num = layer_num.uint + 4
    tcn_activation = tcn_activation.uint
    loss = loss.uint


    learning_rate = learning_rate * (math.exp(-9)) + 0.00001
    dropout_rate = (dropout_rate + 1) * 0.1
    tcn_activation = activation_select[tcn_activation]
    loss = loss_select[loss]
    print('\n windows_length: ', windows_length, ', layer_num: ', layer_num,
          ', learning_rate: ', learning_rate, ', dropout_rate: ', dropout_rate, 'activation:',tcn_activation ,'loss', loss)

#seed 42
optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
# 获取训练集测试集真实的y值
train_y_real = Y_data[windows_length:train_size + windows_length]
test_y_real = Y_data[train_size + windows_length:]
# 根据获得的窗口长度划分数据集
data_x = windowed_dataset(data_x_norm[:-forecast_horizon], feature, windows_length)
data_y = data_y_norm[windows_length:]
train_x = data_x[0:train_size, :, :]
train_y = data_y[0:train_size, :]
test_x = data_x[train_size:, :, :]
test_y = data_y[train_size:, :]

class MTCNAModel(tf.keras.Model):
    def __init__(self, tcn_layer_num, tcn_kernel_size, tcn_filter_num, window_size, forecast_horizon,
                 num_output_time_series, use_bias, kernel_initializer, tcn_dropout_rate, tcn_dropout_format,
                 tcn_activation, tcn_final_activation, tcn_final_stack_activation):
        super(MTCNAModel, self).__init__()

        self.num_output_time_series = num_output_time_series

        # Create stack of TCN layers
        self.lower_tcn = TCNStack(tcn_layer_num, tcn_filter_num, tcn_kernel_size, window_size, use_bias,
                                  kernel_initializer, tcn_dropout_rate, tcn_dropout_format, tcn_activation,
                                  tcn_final_activation, tcn_final_stack_activation)

        self.downsample_att = DownsampleLayerWithAttention(num_output_time_series, window_size, tcn_kernel_size,
                                                           forecast_horizon, kernel_initializer, None)

    def call(self, input_tensor):
        x = self.lower_tcn(input_tensor)
        x, distribution = self.downsample_att([x, input_tensor])
        return x


model = MTCNAModel(layer_num, tcn_kernel_size, tcn_filter_num, windows_length, forecast_horizon, feature,
                   tcn_use_bias, tcn_kernel_initializer, dropout_rate, tcn_dropout_format, tcn_activation,
                   tcn_final_activation, tcn_final_stack_activation)
model.compile(optimizer, loss, metrics=[tf.keras.metrics.RootMeanSquaredError()])
start_time = tf.timestamp()

history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs)

end_time = tf.timestamp()
duration = end_time - start_time
test_pre = model.predict(test_x)
test_pre = np.array(tf.squeeze(test_pre)).reshape(-1, 1)
preprocessor.fit(Y_data)
test_pre_real = preprocessor.inverse_transform(test_pre)

testMAPE = np.mean(np.abs(test_y_real - test_pre_real) / test_y_real)
R2 = r2_score(test_y_real,test_pre_real)
MSE = mean_squared_error(test_y_real,test_pre_real)
RMSE = np.sqrt(mean_squared_error(test_y_real, test_pre_real))
print('RMSE1: ' + str(RMSE))
RMSE = MSE ** 0.5
MAE = mean_absolute_error(test_y_real,test_pre_real)
print("测试集预测平均相对误差：" + str(testMAPE))
print("R2决定系数：" + str(R2))
print('MSE: ' + str(MSE))
print('RMSE2: ' + str(RMSE))
print('MAE: ' + str(MAE))


#训练集预测数据可视化

plt.plot(history.epoch,history.history.get('loss')) #画出随着epoch增大loss的变化图

#测试集训练数据可视化
plt.figure(figsize=(15,6))
bwith = 0.75 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.plot(test_pre_real,label='predict')
plt.plot(test_y_real,label='test')
#plt.plot(test_y_real*(1+0.15),label='15%上限',linestyle='--',color='green')
#plt.plot(test_y_real*(1-0.15),label='15%下限',linestyle='--',color='green')
plt.legend()
plt.show()
