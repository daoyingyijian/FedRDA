# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
import torch
import torch.nn as nn
from flcore.clients.clientRDA import clientRDA
from flcore.servers.serverbase import Server
from threading import Thread
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import copy
class RDA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # self.global_model = None

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRDA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


        self.head = self.clients[0].model.head
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        self.client_data = []
    def tsne_visualization(self):
        for client in self.clients:
            client.send_updates()
            self.client_data.append(client.client_data_up)
        """生成 t-SNE 图"""
        features = []
        labels = []
        client_ids = []

        # 收集所有客户端的数据
        for data in self.client_data:
            features.append(data['features'])
            labels.append(data['labels'])
            client_ids.append(data['client_id'] * torch.ones_like(data['labels']))

        # 将所有客户端的特征和标签拼接在一起
        features = torch.cat(features, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        client_ids = torch.cat(client_ids, dim=0).numpy()

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)
        # 保存 tsne 结果到文件
        np.savez('tsne_results_fedavg.npz', tsne_results=tsne_results, features=features,labels=labels, client_ids=client_ids)

        # 定义用于不同类和客户端的颜色和形状
        colors = plt.cm.get_cmap("tab10", 10)  # 10 种颜色，用于不同的类别
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']  # 10 种形状，用于不同客户端

        plt.figure(figsize=(10, 10))

        # 先绘制实际数据点
        for i in range(10):  # 10个类别
            for client_id in range(10):  # 10个客户端
                indices = (labels == i) & (client_ids == client_id)
                if np.sum(indices) > 0:
                    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                                color=colors(i),
                                marker=markers[client_id % len(markers)],
                                s=20,  # 调整点的大小
                                alpha=0.7)

        # 添加图例，分别显示颜色和形状的含义
        # 图例1：类别对应的颜色
        for i in range(10):
            plt.scatter([], [], color=colors(i), label=f'Class {i}', s=50, edgecolor='none')  # 空图例项

        # 图例2：客户端对应的形状
        for client_id in range(10):
            plt.scatter([], [], color='gray', marker=markers[client_id % len(markers)], label=f'Client {client_id}', s=50)

        plt.legend(loc='best', fontsize='small', ncol=2)  # 调整图例位置和格式
        plt.title('t-SNE of Features (Colored by Class, Shaped by Client)')
        plt.show()
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            #self.send_models(i)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
            self.send_models(i)
            for client in self.selected_clients:
                client.train()
                client.collect_protos()
            self.receive_protos()
            self.receive_models()
            self.global_protos = proto_aggregation(self.uploaded_protos_notrain)
            self.aggregate_parameters()
            self.send_protos()
            # self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            # if i%20==0 and i>1:
            #     self.tsne_visualization()
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def compute_prototype_difference(self):
        """计算客户端与全局模型的差异"""
        client_differences = {}

        for client in self.selected_clients:
            client_id = client.id
            client_protos = client.protos  # 客户端类别原型
            difference_sum = 0
            total_weight = 0

            for label in range(self.num_classes):
                if label in client_protos:
                    client_proto = client_protos[label]
                    global_proto = self.global_protos[label]
                    # 计算差异（L2 距离或余弦相似度）
                    diff = torch.norm(client_proto - global_proto) ** 2
                    # 计算类别权重
                    weight = client.train_samples_by_class[label] / client.train_samples
                    difference_sum += weight * diff
                    total_weight += weight
            # 记录客户端差异
            client_differences[client_id] = difference_sum / total_weight if total_weight > 0 else float('inf')

        return client_differences
    def compute_aggregation_weights(self):

        client_differences=self.compute_prototype_difference()
        # 计算差异权重（差异越小，权重越高）
        inverse_differences = {
            client_id: 1.0 / (diff + 1e-8)
            for client_id, diff in client_differences.items()
        }

        # 归一化差异权重
        total_inverse_diff = sum(inverse_differences.values())
        normalized_diff_weights = {
            client_id: weight / total_inverse_diff
            for client_id, weight in inverse_differences.items()
        }

        # 将差异权重与样本量权重相乘
        combined_weights = {
            client_id: normalized_diff_weights[client_id] * self.uploaded_weights[self.uploaded_ids.index(client_id)]
            for client_id in client_differences.keys()
        }

        # 归一化最终权重
        total_combined_weight = sum(combined_weights.values())
        final_weights = {
            client_id: weight / total_combined_weight
            for client_id, weight in combined_weights.items()
        }

        return final_weights
    def aggregate_parameters(self):
    # 初始化聚合权重
        aggregation_weights = self.compute_aggregation_weights()

        # 创建全局模型的深拷贝作为基础模型
        new_global_model = copy.deepcopy(self.global_model)

        # 遍历全局模型和客户端模型的参数
        # for global_param in new_global_model.parameters():
        #     global_param.data.zero_()  # 将参数初始化为零以便逐步加权累加

        # 按权重更新全局模型
        for client_id, weight in aggregation_weights.items():
            client_model = self.clients[client_id].model
            for global_param, client_param, old_global_param in zip(
                    new_global_model.parameters(), 
                    client_model.parameters(), 
                    self.global_model.parameters()):
                # 按照公式计算：new_global = weight * (client_param - old_global_param)
                global_param.data += weight * (client_param.data - old_global_param.data)

        # 更新全局模型
        self.global_model = copy.deepcopy(new_global_model)
    def send_models(self,R):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model,R)
            #client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_protos_notrain = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos_notrain.append(client.protos)
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos.append((client.protos[cc], y))
            
    def train_head(self):
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        for p, y in proto_loader:
            out = self.head(p)
            loss = self.CEloss(out, y)
            self.opt_h.zero_grad()
            loss.backward()
            self.opt_h.step()
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label
