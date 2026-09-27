import networkx as nx
import numpy as np
import random
import json


# 生成具有周期性边界的方格网络（row * col）   k = 4
def create_periodic_grid_graph_4(rows, cols):
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            # 添加右边的边
            G.add_edge((i, j), (i, (j + 1) % cols))
            # 添加下边的边
            G.add_edge((i, j), ((i + 1) % rows, j))
    return G


# 生成具有周期性边界的方格网络（row * col）   k = 4
def create_periodic_grid_graph_8(rows, cols):
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            # 添加四边的边
            G.add_edge((i, j), (i, (j + 1) % cols))  # 右边
            G.add_edge((i, j), (i, (j - 1) % cols))  # 左边
            G.add_edge((i, j), ((i + 1) % rows, j))  # 下边
            G.add_edge((i, j), ((i - 1) % rows, j))  # 上边
            # 添加对角线的边
            G.add_edge((i, j), ((i + 1) % rows, (j + 1) % cols))  # 右下
            G.add_edge((i, j), ((i + 1) % rows, (j - 1) % cols))  # 左下
            G.add_edge((i, j), ((i - 1) % rows, (j + 1) % cols))  # 右上
            G.add_edge((i, j), ((i - 1) % rows, (j - 1) % cols))  # 左上
    return G


def dict_values(lis, dic):  # 返回dic中以lis为key的value值
    result = []
    for i in lis:
        result.append(dic[i])
    return result


w = 0.01
k = 4
num = 500000  # 计算次数
rows = 10
cols = 10
lam_list = [0.02, 0.05, 0.08] # G1变为G2的速率
mu = 0.02  # G2变为G1的速率
Dg_2 = 0.2
Dr_1 = 0.5
Dr_2 = 0.3


if k == 4:
    G = create_periodic_grid_graph_4(rows, cols)
if k == 8:
    G = create_periodic_grid_graph_8(rows, cols)


for lam in lam_list:
    result = {}
    if lam == 0.02:
        Dg_1_list = np.arange(-0.3, 0.7, 0.1)
    if lam == 0.05:
        Dg_1_list = np.arange(0.2, 0.9, 0.1)
    if lam == 0.08:
        Dg_1_list = np.arange(0.3, 1.1, 0.1)
    for Dg_1 in Dg_1_list:
        num_c = 0  # 纯合作者次数
        pom_1 = [[1, -Dr_1], [1 + Dg_1, 0]]
        pom_2 = [[1, -Dr_2], [1 + Dg_2, 0]]
        for step in range(num):
            # 初始化
            state = {}  # 边状态字典
            next_change = {}  # 边状态改变时间
            strategy = {}  # 玩家策略字典
            for i in G.nodes:
                strategy[i] = True  # True为Defector,False为Cooperator
            invader = random.choice(list(G.nodes()))
            strategy[invader] = False  # 随机选择一个个体采取合作进行入侵
            for i in G.edges:
                if 0 <= random.uniform(0, 1) < 0.5:
                    state[i] = 1
                    next_change[i] = random.expovariate(lam)  # 由G1变为G2的时间
                else:
                    state[i] = 2
                    next_change[i] = random.expovariate(mu)  # 由G2变为G1的时间
            tnow = 0
            t_update = 0  # 每隔1/N时间随机选出一个个体进行死生过程更新策略
            while True:  # 开始每一步的循环
                print("\r" + "lam:{:.2f},Dg_1:{:.2f},step={:.0f}/{:.0f},n_c:{:.0f},num_c:{:d}".format(lam, Dg_1, step, num,
                                                                                           len(strategy) - sum(
                                                                                               strategy.values()),
                                                                                           num_c), end='')
                while t_update < tnow:  # 更新策略
                    death = random.choice(list(G.nodes()))  # 随机选出一个个体死亡
                    death_nei = list(G.neighbors((death)))  # 获取死亡个体的邻居节点列表
                    strategy_death_nei = dict_values(death_nei, strategy)
                    if sum(strategy_death_nei) == 0:  # 死亡个体邻居全是合作者
                        strategy[death] = False
                    elif sum(strategy_death_nei) == len(death_nei):  # 死亡个体邻居全是背叛者
                        strategy[death] = True
                    else:  # 死亡个体邻居既有合作也有背叛
                        # 计算死亡个体邻居中策略的适应度
                        fc = 0  # 合作适应度
                        fd = 0  # 背叛适应度
                        for i in death_nei:  # 遍历死亡个体的邻居节点
                            Ui = 0
                            for interaction in list(G.edges(i)):
                                if interaction not in state:
                                    interaction = (interaction[1], interaction[0])
                                    if state[interaction] == 1:
                                        Ui += pom_1[strategy[i]][strategy[interaction[0]]]
                                    elif state[interaction] == 2:
                                        Ui += pom_2[strategy[i]][strategy[interaction[0]]]
                                else:
                                    if state[interaction] == 1:
                                        Ui += pom_1[strategy[i]][strategy[interaction[1]]]
                                    elif state[interaction] == 2:
                                        Ui += pom_2[strategy[i]][strategy[interaction[1]]]
                            if strategy[i] == False:
                                fc += 1 - w + w * Ui
                            else:
                                fd += 1 - w + w * Ui
                        # 进行策略更新(死生过程)
                        ran = random.uniform(0, 1)  # 生成一个0到1之间的随机数
                        pro_c = fc / (fc + fd)
                        if 0 <= ran < pro_c:
                            strategy[death] = False
                        else:
                            strategy[death] = True
                    t_update += 1 / (rows * cols)
                # 边的状态更新
                changer = min(next_change.items(), key=lambda x: x[1])[0]  # 找出下一个改变状态的边
                tnow = next_change[changer]
                if state[changer] == 1:  # 当前边在进行G1，即将改变状态
                    state[changer] = 2  # 边状态由G1变为进行G2
                    next_change[changer] += random.expovariate(mu)  # 边下一次变为进行G1的时间
                elif state[changer] == 2:  # 当前边在进行G2，即将改变状态
                    state[changer] = 1  # 边状态由G2变为进行G1
                    next_change[changer] += random.expovariate(lam)  # 边下一次变为进行G2的时间

                if len(strategy) - sum(strategy.values()) == 0:  # 如果所有个体都是背叛者
                    break  # 结束模拟
                if len(strategy) - sum(strategy.values()) == rows * cols:  # 如果所有个体都是合作者
                    # 更新纯合作者次数
                    num_c += 1
                    # print(num_c)
                    break
        result['Dg_1: {:.2f}'.format(Dg_1)] = num_c / num