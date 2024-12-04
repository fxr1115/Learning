import numpy as np

from utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        data_processed, features_mean, features_deviation = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        number_features = self.data.shape[1]
        self.theta = np.zeros((number_features, 1))

    @staticmethod
    def hypothesis(data, theta):
        prediction = np.dot(data, theta)
        return prediction
    
    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法，并且是矩阵计算
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        # 注意：点积的乘法！！
        # theta是列向量
        theta = theta - alpha * (1/num_examples) * np.dot(delta.T, self.data).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(data, self.theta) - labels
        cost = np.dot(delta.T, delta) / (2 * num_examples)
        
        # 得到的cost是一个 1*1 的二维数组，形如 [[value]]
        return cost[0][0] 
    
    def gradient_descent(self, alpha, num_iterations=500):
        """
        实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        # 使用 _
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        
        return cost_history
            
    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history   

    # 测试test
    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练好的参数模型去预测得到回归值结果
        """
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions      