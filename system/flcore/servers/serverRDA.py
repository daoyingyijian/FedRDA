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
        self.set_slow_clients()
        self.set_clients(clientRDA)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.head = self.clients[0].model.head
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        self.client_data = []
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
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        self.save_results()

    def compute_prototype_difference(self):
        client_differences = {}
        for client in self.selected_clients:
            client_id = client.id
            client_protos = client.protos 
            difference_sum = 0
            total_weight = 0
            for label in range(self.num_classes):
                if label in client_protos:
                    client_proto = client_protos[label]
                    global_proto = self.global_protos[label]
                    diff = torch.norm(client_proto - global_proto) ** 2
                    weight = client.train_samples_by_class[label] / client.train_samples
                    difference_sum += weight * diff
                    total_weight += weight
            client_differences[client_id] = difference_sum / total_weight if total_weight > 0 else float('inf')

        return client_differences
    def compute_aggregation_weights(self):

        client_differences=self.compute_prototype_difference()
        inverse_differences = {
            client_id: 1.0 / (diff + 1e-8)
            for client_id, diff in client_differences.items()
        }
        total_inverse_diff = sum(inverse_differences.values())
        normalized_diff_weights = {
            client_id: weight / total_inverse_diff
            for client_id, weight in inverse_differences.items()
        }
        combined_weights = {
            client_id: normalized_diff_weights[client_id] * self.uploaded_weights[self.uploaded_ids.index(client_id)]
            for client_id in client_differences.keys()
        }
        total_combined_weight = sum(combined_weights.values())
        final_weights = {
            client_id: weight / total_combined_weight
            for client_id, weight in combined_weights.items()
        }

        return final_weights
    def aggregate_parameters(self):
        aggregation_weights = self.compute_aggregation_weights()
        new_global_model = copy.deepcopy(self.global_model)
        for client_id, weight in aggregation_weights.items():
            client_model = self.clients[client_id].model
            for global_param, client_param, old_global_param in zip(
                    new_global_model.parameters(), 
                    client_model.parameters(), 
                    self.global_model.parameters()):
                global_param.data += weight * (client_param.data - old_global_param.data)
        self.global_model = copy.deepcopy(new_global_model)
    def send_models(self,R):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model,R)
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
