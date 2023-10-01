import json
import random
import networkx as nx
import pprint
from gurobipy import *
from pulp import *

path = '../Data/'
SFCType = [["FW", "LB"], ["IDS", "FW", "NAT", "Router"], ["VPN", "TM", "FW", "LB"], ["IDS", "TM", "LB"], ["NAT", "FW", "IDS", "VPN"]]
NFType = ["FW", "LB", "IDS", "NAT", "Router", "VPN", "TM"]
SFCTypeNum = len(SFCType)
ProgrammableSwitchPrice = 7500
SmartNicPrice = 2738


class OurDeploymentMethod(object):
    def __init__(self, requestList, switch_list):
        self.requestList = requestList 
        self.switch_list = switch_list 
        self.aaa = 0

    #If there is still unprocessed traffic, return True, otherwise return False
    def is_unprocessed_traffic(self, g_f):
        self.aaa += 1
        count = 0
        for NF in NFType:
            count += len(g_f[NF])
        # print("aaa, count: ", self.aaa, count)
        if self.aaa >= 500:
            return False
        if count != 0:
            return True
        else:
            return False


    def KP(self, switch, U_vf):
        offloaded_NF_list = [] 
        A_vf = {}  
        a_vf = {}
        for NF in NFType:
            A_vf[NF] = set()
            a_vf[NF] = 0

        for NF in NFType:
            request_dic = {}
            for request_id in U_vf[switch][NF]:
                request_dic[request_id] = self.requestList[request_id]["traffic_size"]
            deordered_request_list = sorted(request_dic.items(), key=lambda x: x[1], reverse=True)
            load = 0
            for request in deordered_request_list:
                request_id = request[0]
                traffic_size = request[1]
                load += traffic_size
                if load <= NFCapacity:
                    A_vf[NF].add(request_id)
                else:
                    break

            for request_id in A_vf[NF]:
                traffic_size = self.requestList[request_id]["traffic_size"]
                a_vf[NF] += traffic_size
        #print("a_vf: ", a_vf)

        a_vf_unit_profit = {}
        for NF in NFType:
            a_vf_unit_profit[NF] = a_vf[NF] / NFOffloadCost[NF]
        deordered_a_vf_unit_profit = sorted(a_vf_unit_profit.items(), key=lambda x: x[1], reverse=True)
        cost = 0
        for NF in deordered_a_vf_unit_profit:
            cost += NFOffloadCost[NF[0]]
            if cost <= ProgrammableResourceCapacity:
                offloaded_NF_list.append(NF[0])

        profit = 0
        for NF in NFType:
            profit += a_vf[NF]

        return offloaded_NF_list, A_vf, profit

    def deploymentAlgorithm(self):
        # step1: select switches to update
        count = 0 #用于debug
        programmable_switch_list = [] 
        P_v = {}

       
        g_f = {} 
        for NF in NFType:
            g_f[NF] = set()
        requestLength = len(self.requestList)
        for i in range(requestLength):
            SFC = requestList[i]["SFC"]
            for NF in SFC:
                g_f[NF].add(i)
        #pprint.pprint(g_f)

        U_vf = {}
        for switch in self.switch_list:
            U_vf[switch] = {}
            for NF in NFType:
                U_vf[switch][NF] = set()
        for i in range(requestLength):
            SFC = requestList[i]["SFC"]
            path = requestList[i]["path"][1:-1] 
            for switch in path:
                for NF in SFC:
                    U_vf[switch][NF].add(i)

        while True:
            ans = self.is_unprocessed_traffic(g_f)
            if not ans:
                break

            candidate_switch_list = [] 
            for switch in self.switch_list:
                if switch not in programmable_switch_list:
                    candidate_switch_list.append(switch)

            best_profit = 0
            best_offloaded_NF_list = []
            best_A_vf = []
            best_switch = None
            for switch in candidate_switch_list:
                offloaded_NF_list, A_vf, profit = self.KP(switch, U_vf)
                if profit > best_profit:
                    best_profit = profit
                    best_offloaded_NF_list = offloaded_NF_list
                    best_A_vf = A_vf
                    best_switch = switch
            if best_switch != None:
                programmable_switch_list.append(best_switch)
                P_v[best_switch] = best_offloaded_NF_list
            #print("best_switch, best_offloaded_NF_list, best_profit, best_A_vf: ", best_switch, best_offloaded_NF_list, best_profit, best_A_vf)
            # print("best_A_vf: ", best_A_vf)

            #update request set
            for NF_name in best_offloaded_NF_list:
                #print("before: ", g_f[NF])
                # temp = g_f[NF_name]
                g_f[NF_name] = g_f[NF_name] - best_A_vf[NF_name]
                # if temp != g_f[NF_name]:
                #     print("updated!!")
                #print("after: ", g_f[NF])
                #pprint.pprint(g_f)

            # candidate_switch_list.remove(best_switch)
            for switch in candidate_switch_list:
                for NF_name in best_offloaded_NF_list:
                    U_vf[switch][NF_name] = U_vf[switch][NF_name] - best_A_vf[NF_name]

 
        server_programmable_switch_list = [[]]
        programmable_switch_memory_cost = {}
        for programmable_switch in programmable_switch_list:
            programmable_switch_memory_cost[programmable_switch] = 0
            for NF in P_v[programmable_switch]:
                programmable_switch_memory_cost[programmable_switch] += NFMemoryCost[NF]
        # pprint.pprint(programmable_switch_memory_cost)

        deordered_programmable_switch_memory_cost = sorted(programmable_switch_memory_cost.items(), key=lambda x: x[1], reverse=True)
        for programmable_switch in deordered_programmable_switch_memory_cost:
            memory_cost = programmable_switch[1]
            programmable_switch_name = programmable_switch[0]

            flag = 0  
            server_num = len(server_programmable_switch_list)
            for server_id in range(server_num):
                programmable_list = server_programmable_switch_list[server_id]
                # print(programmable_list)
                memory_load = 0
                #print(programmable_list)
                for programmable in programmable_list:
                    memory_load += programmable_switch_memory_cost[programmable]
                if memory_load + memory_cost <= ServerMemoryCapacity:
                    server_programmable_switch_list[server_id].append(programmable_switch_name)
                    flag = 1
                    break

            if flag == 0:
                server_programmable_switch_list.append([])
                server_programmable_switch_list[-1].append(programmable_switch_name)

        price = len(programmable_switch_list) * ProgrammableSwitchPrice + len(server_programmable_switch_list) * SmartNicPrice
        programmable_switch_per_server_price = len(programmable_switch_list) * (ProgrammableSwitchPrice + SmartNicPrice)
        # print("len!!!: ", len(programmable_switch_list))

        return programmable_switch_list, P_v, price, programmable_switch_per_server_price


class RBNS(object):
    def __init__(self, a, k, q, normal_flow_list, nf_nodes, tenant_SFC_list):
        self.a = a
        self.k = k
        self.q = q
        self.normal_flow_list = normal_flow_list
        self.nf_nodes = nf_nodes
        self.tenant_SFC_list = tenant_SFC_list
        self.tenant_gran_flow_list = self.return_tenant_gran_flow_list()

    def return_tenant_gran_flow_list(self):
        tenant_gran_traffic_size_list = [0] * self.a
        for flow in self.normal_flow_list:
            tenant_id = flow["tenant_id"]
            traffic_size = flow["traffic_size"]
            tenant_gran_traffic_size_list[tenant_id] += traffic_size

        tenant_gran_flow_list = []
        for tenant_id in range(self.a):
            flow = {}
            flow["tenant_id"] = tenant_id
            flow["SFC"] = self.tenant_SFC_list[tenant_id]
            flow["traffic_size"] = tenant_gran_traffic_size_list[tenant_id]
            tenant_gran_flow_list.append(flow)
        return tenant_gran_flow_list

    def solve_lp(self):
        x_name = []
        for tenant_id in range(self.a):
            flow = self.tenant_gran_flow_list[tenant_id]
            SFC = flow["SFC"]
            SFC_len = len(SFC)
            name_temp = []
            for NF_id in range(SFC_len):
                NF = SFC[NF_id]
                NF_instance_list = self.nf_nodes[NF]
                instance_list_len = len(NF_instance_list)
                name_temp.append([str(tenant_id) + "_" + str(NF_id) + "_" + str(instance_id) for instance_id in range(instance_list_len)])
            x_name.append(name_temp)

        x = []
        for tenant_id in range(self.a):
            flow = self.tenant_gran_flow_list[tenant_id]
            SFC = flow["SFC"]
            SFC_len = len(SFC)
            name_temp = []
            for NF_id in range(SFC_len):
                NF = SFC[NF_id]
                NF_instance_list = self.nf_nodes[NF]
                instance_list_len = len(NF_instance_list)
                for instance_id in range(instance_list_len):
                    x.append(x_name[tenant_id][NF_id][instance_id])

        y_name = copy.deepcopy(x_name)
        y = copy.deepcopy(x)

        prob = LpProblem("first step: middlebox assignment", LpMinimize)
        lambda_var = LpVariable("lambda", lowBound=0, cat=LpContinuous)
        x_vars = LpVariable.dicts("x", x, lowBound=0, upBound=1, cat=LpContinuous)
        y_vars = LpVariable.dicts("y", y, lowBound=0, upBound=1, cat=LpContinuous)

  
        prob += lambda_var

        
        for tenant_id in range(self.a):
            flow = self.tenant_gran_flow_list[tenant_id]
            SFC = flow["SFC"]
            SFC_len = len(SFC)
            for NF_id in range(SFC_len):
                NF = SFC[NF_id]
                NF_instance_list = self.nf_nodes[NF]
                instance_list_len = len(NF_instance_list)
                prob += lpSum([x_vars[x_name[tenant_id][NF_id][instance_id]] for instance_id in range(instance_list_len)]) == 1

       
        for tenant_id in range(self.a):
            flow = self.tenant_gran_flow_list[tenant_id]
            SFC = flow["SFC"]
            SFC_len = len(SFC)
            for NF_id in range(SFC_len):
                NF = SFC[NF_id]
                NF_instance_list = self.nf_nodes[NF]
                instance_list_len = len(NF_instance_list)
                for instance_id in range(instance_list_len):
                    prob += x_vars[x_name[tenant_id][NF_id][instance_id]] <= y_vars[y_name[tenant_id][NF_id][instance_id]]

    
        for tenant_id in range(self.a):
            SFC = self.tenant_gran_flow_list[tenant_id]["SFC"]
            SFC_len = len(SFC)
            for NF_id in range(SFC_len):
                NF_instance_list = self.nf_nodes[SFC[NF_id]]
                instance_list_len = len(NF_instance_list)
                prob += lpSum([y_vars[y_name[tenant_id][NF_id][instance_id]]  for instance_id in range(instance_list_len)]) <= self.k

        
        NF_tenant_dic = {}
        for NF in NF_Set:
            NF_tenant_dic[NF] = []
        for tenant_id in range(self.a):
            SFC = self.tenant_gran_flow_list[tenant_id]["SFC"]
            for NF in SFC:
                NF_tenant_dic[NF].append(tenant_id)

        NF_SFC_index = {}
        for NF in NF_Set:
            NF_SFC_index[NF] = [100] * self.a 
            for tenant_id in range(self.a):
                SFC = self.tenant_gran_flow_list[tenant_id]["SFC"]
                if NF in SFC:
                    NF_index = SFC.index(NF)
                    NF_SFC_index[NF][tenant_id] = NF_index

        NF_len = len(NF_Set)
        for NF_id in range(NF_len):
            NF = NF_Set[NF_id]
            NF_instance_list = self.nf_nodes[NF]
            instance_list_len = len(NF_instance_list)
            tenant_satisfied_list = NF_tenant_dic[NF]
            NF_SFC_index_list = NF_SFC_index[NF]
            for instance_id in range(instance_list_len):
                prob += lpSum([y_vars[y_name[tenant_id][NF_SFC_index_list[tenant_id]][instance_id]] for tenant_id in tenant_satisfied_list]) <= self.q

       
        traffic_list = []
        for tenant_id in range(self.a):
            traffic_list.append(self.tenant_gran_flow_list[tenant_id]["traffic_size"])

        for NF_id in range(NF_len):
            NF = NF_Set[NF_id]
            NF_instance_list = self.nf_nodes[NF]
            NF_SFC_index_list = NF_SFC_index[NF]
            instance_list_len = len(NF_instance_list)
            for instance_id in range(instance_list_len):
                tenant_satisfied_list = NF_tenant_dic[NF]
                prob += lpSum([x_vars[x_name[tenant_id][NF_SFC_index_list[tenant_id]][instance_id]] * traffic_list[tenant_id] for tenant_id in tenant_satisfied_list]) <= lambda_var * NF_Capacity[NF]

       
        prob.solve(GUROBI_CMD())
        print("Status: ", LpStatus[prob.status])
        print("optimal value: ", value(prob.objective))

        result = {}
        for v in prob.variables():
            result[v.name] = v.varValue

        return result

class FatTree(object):
    def __init__(self):
        self.topo = nx.Graph()
        self.host_list = []
        self.switch_list = []

    def buildTopo(self, n=8):
        """Standard fat tree topology
        n: number of pods
        total n^3/4 hosts
        """
        num_of_hosts_per_edge_switch = n // 2
        num_of_edge_switches = n // 2
        num_of_aggregation_switches = num_of_edge_switches
        num_of_core_switches = int((n / 2) * (n / 2))

        # generate topo pod by pod
        for i in range(n):
            for j in range(num_of_edge_switches):
                self.topo.add_node("Pod {} edge switch {}".format(i, j))
                self.switch_list.append("Pod {} edge switch {}".format(i, j))
                self.topo.add_node("Pod {} aggregation switch {}".format(i, j))
                self.switch_list.append("Pod {} aggregation switch {}".format(i, j))
                for k in range(num_of_hosts_per_edge_switch):
                    self.topo.add_node("Pod {} edge switch {} host {}".format(
                        i, j, k))
                    self.host_list.append("Pod {} edge switch {} host {}".format(
                        i, j, k))
                    self.topo.add_edge(
                        "Pod {} edge switch {}".format(i, j),
                        "Pod {} edge switch {} host {}".format(i, j, k))

        # add edge among edge and aggregation switch within pod
        for i in range(n):
            for j in range(num_of_aggregation_switches):
                for k in range(num_of_edge_switches):
                    self.topo.add_edge("Pod {} aggregation switch {}".format(i, j),
                                "Pod {} edge switch {}".format(i, k))

        # add edge among core and aggregation switch
        num_of_core_switches_connected_to_same_aggregation_switch = num_of_core_switches // num_of_aggregation_switches
        for i in range(num_of_core_switches):
            self.topo.add_node("Core switch {}".format(i))
            self.switch_list.append("Core switch {}".format(i))
            aggregation_switch_index_in_pod = i // num_of_core_switches_connected_to_same_aggregation_switch
            for j in range(n):
                self.topo.add_edge(
                    "Core switch {}".format(i),
                    "Pod {} aggregation switch {}".format(
                        j, aggregation_switch_index_in_pod))

        self.topo.name = 'fattree'


