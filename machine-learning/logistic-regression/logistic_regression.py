import numpy as np

from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid

class logistic_regression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        data_processed, features_mean, features_deviation = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        
        self.data = data_processed
        self.labels = labels
        # 类别
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = self.unique_labels.shape[0]
        self.theta = np.zeros((num_unique_labels, num_features)) 

    @staticmethod
    def hypothesis(data, theta_):        
        predictions = sigmoid(np.dot(data, theta_))
        return predictions
    
    @staticmethod
    def cost_function(data, labels, theta_):
        num_examples = data.shape[0]
        predictions = logistic_regression.hypothesis(data, theta_)
        # y_is_1 = labels[labels == 1].T.dot(np.log(predictions[labels == 1]))
        # y_is_0 = (1- labels[labels == 0]).T.dot(np.log(1 - predictions[labels == 0]))
        y_is_set_cost = np.sum(np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.sum(np.log(1 - predictions[labels == 0]))
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def gradient_step(data, labels, theta_):
        num_examples = data.shape[0]
        predictions = logistic_regression.hypothesis(data, theta_)
        label_off = predictions - labels
        gradients = (1 / num_examples) * np.dot(data.T, label_off) # 维度: 1*num_features
        return gradients.flatten() ##
          
    @staticmethod
    def gradient_descent(data, labels, initial_theta, max_iter):
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
                    lambda current_theta:logistic_regression.cost_function(data, labels, current_theta.reshape(-1, 1)),
                    # 初始化权重参数
                    initial_theta.flatten(),
                    # 共轭梯度法
                    method = 'CG',
                    # 梯度下计算公式
                    jac = lambda current_theta:logistic_regression.gradient_step(data, labels, current_theta.reshape(num_features, 1)),
                    # 记录结果
                    callback = lambda current_theta:cost_history.append(logistic_regression.cost_function(data, labels, current_theta.reshape(num_features, 1))),
                    options = {'maxiter' : max_iter},
                )
        if not result.success:
            raise ArithmeticError('Can not minimize cost function' + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history
        
    def train(self, max_iter):
        cost_histories = []
        num_feartures = self.data.shape[1]
        # 有好几个分类器
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index].reshape(-1, 1))
            # 当前类别，就分为两类
            current_labels = (self.labels == unique_label).astype(float)
            current_theta, cost_history = logistic_regression.gradient_descent(self.data, current_labels, current_initial_theta, max_iter)
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)
        return self.theta, cost_histories

    def predict(self, data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree,self.sinusoid_degree, self.normalize_data)[0]
        pro = logistic_regression.hypothesis(data_processed, self.theta.T)
        # 最大值的索引
        max_prob_index = np.argmax(pro, axis=1) 

        #使用的是NumPy的高级索引
        array_label = np.tile(self.unique_labels, (data.shape[0], 1))
        class_prediction = array_label[np.arange(data_processed.shape[0]), max_prob_index]
        
        # class_prediction = np.empty(max_prob_index.shape, dtype=object)
        # for index, label in enumerate(self.unique_labels):
        #     class_prediction[max_prob_index == index] = label
        return class_prediction.reshape(-1, 1)