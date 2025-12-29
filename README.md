机器学习算法通常按照**学习方式**可分为以下几大类：

## 一、监督学习（Supervised Learning）

需要标注数据，学习输入到输出的映射关系。

**分类算法**：逻辑回归（Logistic Regression）、支持向量机（SVM）、K近邻（KNN）、朴素贝叶斯（Naive Bayes）、决策树（Decision Tree）、随机森林（Random Forest）、梯度提升树（GBDT/XGBoost/LightGBM/CatBoost）、神经网络/深度学习（Neural Networks）

**回归算法**：线性回归（Linear Regression）、岭回归（Ridge Regression）、Lasso回归、弹性网络（Elastic Net）、支持向量回归（SVR）、决策树回归、随机森林回归、梯度提升回归、神经网络回归

---

## 二、无监督学习（Unsupervised Learning）

不需要标签，发现数据内在结构。

**聚类算法**：K-Means、层次聚类（Hierarchical Clustering）、DBSCAN、Mean Shift、高斯混合模型（GMM）、谱聚类（Spectral Clustering）、OPTICS、BIRCH

**降维算法**：主成分分析（PCA）、线性判别分析（LDA）、t-SNE、UMAP、独立成分分析（ICA）、因子分析（Factor Analysis）、奇异值分解（SVD）

**关联规则**：Apriori、FP-Growth

**异常检测**：孤立森林（Isolation Forest）、One-Class SVM、LOF（局部离群因子）

---

## 三、半监督学习（Semi-Supervised Learning）

结合少量标注数据和大量无标注数据。

常见方法包括：自训练（Self-Training）、协同训练（Co-Training）、标签传播（Label Propagation）、图神经网络方法、伪标签（Pseudo-Labeling）

---

## 四、强化学习（Reinforcement Learning）

智能体通过与环境交互学习策略。

**基于值函数**：Q-Learning、DQN（Deep Q-Network）、SARSA、Double DQN、Dueling DQN

**基于策略梯度**：REINFORCE、Actor-Critic、A2C/A3C、PPO（Proximal Policy Optimization）、TRPO、SAC（Soft Actor-Critic）、DDPG、TD3

**基于模型**：Model-Based RL、Dyna-Q、World Models

---

## 五、集成学习（Ensemble Learning）

组合多个模型提升性能。

包括：Bagging（如随机森林）、Boosting（如AdaBoost、GBDT、XGBoost）、Stacking、Voting

---

## 六、深度学习（Deep Learning）

作为独立的大类，涵盖多种架构。

**前馈网络**：多层感知机（MLP）

**卷积神经网络**：CNN、ResNet、VGG、Inception、EfficientNet

**循环神经网络**：RNN、LSTM、GRU

**注意力机制/Transformer**：Transformer、BERT、GPT系列、ViT

**生成模型**：GAN（生成对抗网络）、VAE（变分自编码器）、Diffusion Models、Flow-based Models

**图神经网络**：GCN、GAT、GraphSAGE

---


机器学习算法通常按照**任务类型**可分为以下几大类：

## 一、分类算法（Classification）

用于预测离散类别标签。

### 线性模型
逻辑回归（Logistic Regression）、线性判别分析（LDA）、感知机（Perceptron）

### 基于距离
K近邻（KNN）、最近质心分类器（Nearest Centroid）

### 概率模型
朴素贝叶斯（Gaussian/Multinomial/Bernoulli Naive Bayes）、贝叶斯网络（Bayesian Network）、隐马尔可夫模型（HMM）

### 支持向量机
SVM（线性核/多项式核/RBF核/Sigmoid核）

### 决策树系列
决策树（ID3/C4.5/CART）、随机森林（Random Forest）、极端随机树（Extra Trees）

### 集成/提升方法
AdaBoost、GBDT、XGBoost、LightGBM、CatBoost、Stacking、Voting Classifier

### 神经网络/深度学习
多层感知机（MLP）、卷积神经网络（CNN）、循环神经网络（RNN/LSTM/GRU）、Transformer、ResNet、BERT（文本分类）、ViT（图像分类）

### 其他
二次判别分析（QDA）、规则学习（RIPPER/OneR）、遗传算法分类器

---

## 二、回归算法（Regression）

用于预测连续数值。

### 线性模型
线性回归（Linear Regression）、岭回归（Ridge Regression）、Lasso回归、弹性网络（Elastic Net）、贝叶斯线性回归（Bayesian Linear Regression）

### 多项式与曲线拟合
多项式回归（Polynomial Regression）、样条回归（Spline Regression）、LOESS/LOWESS（局部加权回归）

### 基于距离
K近邻回归（KNN Regressor）

### 支持向量机
支持向量回归（SVR，线性核/RBF核/多项式核）

### 决策树系列
决策树回归（CART）、随机森林回归、极端随机树回归（Extra Trees Regressor）

### 集成/提升方法
AdaBoost Regressor、GBDT Regressor、XGBoost Regressor、LightGBM Regressor、CatBoost Regressor、Stacking Regressor

### 神经网络/深度学习
多层感知机回归（MLP Regressor）、卷积神经网络回归、循环神经网络回归（RNN/LSTM/GRU）、Transformer回归、深度残差网络回归

### 概率/贝叶斯模型
高斯过程回归（Gaussian Process Regression）、贝叶斯回归

### 其他
分位数回归（Quantile Regression）、保序回归（Isotonic Regression）、Huber回归（鲁棒回归）、RANSAC回归、Theil-Sen回归

---

## 三、既可分类也可回归的算法

| 算法 | 分类版本 | 回归版本 |
|------|----------|----------|
| K近邻 | KNeighborsClassifier | KNeighborsRegressor |
| 决策树 | DecisionTreeClassifier | DecisionTreeRegressor |
| 随机森林 | RandomForestClassifier | RandomForestRegressor |
| 梯度提升 | GradientBoostingClassifier | GradientBoostingRegressor |
| XGBoost | XGBClassifier | XGBRegressor |
| LightGBM | LGBMClassifier | LGBMRegressor |
| CatBoost | CatBoostClassifier | CatBoostRegressor |
| SVM | SVC | SVR |
| 神经网络 | MLPClassifier | MLPRegressor |
| AdaBoost | AdaBoostClassifier | AdaBoostRegressor |

---