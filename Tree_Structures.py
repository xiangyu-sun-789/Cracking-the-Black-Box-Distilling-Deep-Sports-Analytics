import numpy as np, sys
import math
from scipy.stats import ttest_ind
import copy
import time
from sklearn.mixture import GaussianMixture
from sympy.solvers import solve
from sympy import Symbol    
from numpy.linalg import lstsq
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

NodeSplit = 0
NodeLeaf = 1
NodeFringe = 2
ActionDimension = -1
AbsorbAction = 5
HOME = 0
AWAY = 1
FEATURE_NAME_DICT = {'position': 0, 'velocity': 1}

class Util:
    @staticmethod
    def isEventPaddings(event_number, inst, problem):
        event_is_padding = False
        # event 0 has no paddings
        if event_number >= 1:
            if problem.__class__.__name__ == 'SPG_ice_hockey':
                # 39 = 12 state features + 27 one-hot actions
                # 10th state feature is home (index 9)
                # 11th state feature is away (index 10)
                home_index = (9 - event_number) * 39 + 9
                away_index = (9 - event_number) * 39 + 10

            elif problem.__class__.__name__ == 'SPG_soccer':
                # 61 = 18 state features + 43 one-hot actions
                # 17th state feature is home (index 16)
                # 18th state feature is away (index 17)
                home_index = (9 - event_number) * 61 + 16
                away_index = (9 - event_number) * 61 + 17

            # if the instance has no values on the split candidate feature but paddings instead, its home and aways values are both 0s.
            event_is_padding = inst.currentObs[home_index] == 0 and inst.currentObs[away_index] == 0

        return event_is_padding

class Model_Tree:
    def __init__(self, gamma, n_actions, dim_sizes, dim_names, problem, minSplitInstances, max_depth, Q_value_difference_significant_level, min_Q_difference_threshold, training_epochs, variance_reduction_prune, number_of_gaps, split_method, model_type, pruning_method, pruning_lamb, max_back_depth=1, training_mode=''):

        # LR = linear_regression.LinearRegression()
        # weight = LR.weight_initialization()
        # bias = LR.bias_initialization()
        self.problem = problem

        self.node_id_count = 0
        self.root = Tree_Node(self.genId(), NodeLeaf, None, n_actions, 1, self.problem)
        self.n_actions = n_actions
        self.max_back_depth = max_back_depth
        self.gamma = gamma
        self.instances = []
        self.n_dim = len(dim_sizes)
        self.dim_sizes = dim_sizes
        self.dim_names = dim_names
        self.minSplitInstances = minSplitInstances
        self.nodes = {self.root.idx: self.root}
        self.training_mode = training_mode
        self.game_number = None
        self.max_depth = max_depth
        self.training_epochs = training_epochs
        self.Q_value_difference_significant_level = Q_value_difference_significant_level
        self.min_Q_difference_threshold = min_Q_difference_threshold
        self.variance_reduction_prune = variance_reduction_prune
        self.split_method = split_method
        self.model_type = model_type
        self.number_of_gaps = number_of_gaps
        self.pruning_method = pruning_method
        self.pruning_lamb = pruning_lamb

        return

    def descendantHasGoodVarianceReduction(self, variance_reduction_prune, node):
        if node.nodeType == NodeSplit:
            if node.distinction.split_criterion_value >= variance_reduction_prune:
                return True
            else:
                for child in node.children:
                    result = self.descendantHasGoodVarianceReduction(variance_reduction_prune, child)

                    if result == True:
                        return True
                    
                return False

        elif node.nodeType == NodeLeaf:
            return None
        else:
            raise ValueError(('Unsupported tree nodeType:{0}').format(node.nodeType))

    def prune_intermediate_node(self, node):
        # node is already a split node

        child_node_id = []

        average_Q_list = []
        instances_list = []

        for child in node.children:
            child_node_id.append(child.idx)

            if child.nodeType == NodeSplit:
                average_Q, instances = self.prune_intermediate_node(child)

                average_Q_list.append(average_Q)
                instances_list.append(instances)
            elif child.nodeType == NodeLeaf:
                average_Q, instances = child.qValues[0], child.instances

                average_Q_list.append(average_Q)
                instances_list.append(instances)
            else:
                raise ValueError(('Unsupported tree nodeType:{0}').format(node.nodeType))
        
        node.instances = []
        for instances in instances_list:
            node.instances.extend(instances)

        self.pick_model_and_train(node)

        # calculate merged values for the node
        total_number_instances = float(len(node.instances))
        average_Q = 0
        for i, instances in enumerate(instances_list):
            child_number_instances = float(len(instances))
            weight = child_number_instances / total_number_instances

            average_Q = average_Q + average_Q_list[i] * weight
        
        node.qValues[0] = average_Q

        # remove the child nodes from tree
        for i, node_from_tree in self.nodes.items():
            if node_from_tree.idx in child_node_id:
                del self.nodes[node_from_tree.idx]

        # remove the child nodes from the node
        # make the node a leaf
        # remove split feature
        node.children = []
        node.nodeType = NodeLeaf
        node.distinction = None

        return node.qValues[0], node.instances

    def prune_leaf_nodes(self):
        for i, node in self.nodes.items():
            if node.nodeType == NodeLeaf:
                parent_node = node.parent

                # if one of the other child nodes is not a leaf node, then do not prune
                child_not_leaf = False
                for child_node in parent_node.children:
                    if child_node.nodeType != NodeLeaf:
                        child_not_leaf = True

                if child_not_leaf == True:
                    continue
                
                child_node_id = []
                instances_list = []
                total_number_of_instances = 0
                total_children_se = 0
                total_children_weights = 0
                for child_node in parent_node.children:
                    child_node_id.append(child_node.idx)
                    instances_list.append(child_node.instances)

                    child_number_of_instances = len(child_node.instances)
                    total_number_of_instances = total_number_of_instances + child_number_of_instances

                    child_se = pow(child_node.rmse, 2) * child_number_of_instances
                    total_children_se = total_children_se + child_se
                    if self.pruning_method == 2:
                        total_children_weights = total_children_weights + sum(abs(child_node.weight))
                    if self.pruning_method == 3:
                        total_children_weights = total_children_weights + len(child_node.weight)
                
                parent_se = pow(parent_node.rmse, 2) * total_number_of_instances
                if self.pruning_method == 2:
                    parent_weights = sum(abs(parent_node.weight))
                if self.pruning_method == 3:
                    parent_weights = len(parent_node.weight)

                # objective function is squared error + penalty on weights 
                obj_parent = parent_se + self.pruning_lamb * parent_weights
                obj_total_children = total_children_se + self.pruning_lamb * total_children_weights

                # start pruning 
                if (obj_parent <= obj_total_children):
                    # print('prune nodes: {}'.format(child_node_id))

                    # average Q, RMSE, weights are already calculated when building the tree 

                    # add instances back to the parent node 
                    parent_node.instances = []
                    for instances in instances_list:
                        parent_node.instances.extend(instances)

                    # remove the child nodes from tree
                    for i, node_from_tree in self.nodes.items():
                        if node_from_tree.idx in child_node_id:
                            del self.nodes[node_from_tree.idx]

                    # remove the child nodes from the node
                    # make the node a leaf
                    # remove split feature
                    parent_node.children = []
                    parent_node.nodeType = NodeLeaf
                    parent_node.distinction = None

                    # return false which do not break while loop
                    return False

        # if no more leaf nodes to prune, then return true to break while loop
        return True

    def calculate_average_RMSE(self, total_instances_number):
        total_instances_number = float(total_instances_number)
        average_RMSE = 0
        for i, node in self.nodes.items():
            if node.nodeType == NodeLeaf:
                node_instances_number = len(node.instances)
                average_RMSE = average_RMSE + (node_instances_number / total_instances_number) * node.rmse

        return average_RMSE

    def RMSE_on_test_set(self, x_test, y_test):
        '''
        calculate RMSE of the whole tree on held-out test set
        '''
        y_predicted = []
        for i in range(len(x_test)):
            node = self.root
            while node.nodeType != NodeLeaf:
                data_instance = Instance(0, x_test[i], 0, y_test[i])
                feature_index = node.distinction.dimension_index

                event_number = int(node.distinction.dimension_name[-1])
                event_is_padding = Util.isEventPaddings(event_number, data_instance, self.problem)
                if event_is_padding == True:
                    if len(node.children) < 3:
                        # if the training set did not have a padded data but testing data has, inherit the parent node as the 3rd child node for paddings 
                        idx = self.genId() # ID/Index will be assigned to the new child Node
                        print('inherited node ID for paddings: {}'.format(idx))
                        child_node = Tree_Node(idx, NodeLeaf, node, self.n_actions, node.depth + 1, self.problem)
                        self.nodes[idx] = child_node
                        node.children.append(child_node)
                        child_node.weight = node.weight
                        child_node.bias = node.bias
                    
                    node = node.children[2]
                    
                else:
                    if node.distinction.iscontinuous == True:
                        if data_instance.currentObs[feature_index] <= node.distinction.continuous_divide_value:
                            node = node.children[0]
                        else:
                            node = node.children[1]
                    else:
                        if data_instance.currentObs[feature_index] == node.distinction.discrete_divide_value:
                            node = node.children[0]
                        else:
                            node = node.children[1]
            
            if node.nodeType == NodeLeaf:
                if self.model_type == 1: # linear model 
                    x_features = np.append(data_instance.currentObs, [1.0]) # add [1.0] as bias term
                    prediciton = x_features.dot(node.weight.T)
                    y_predicted.append(prediciton)
                elif self.model_type == 2: # average constant
                    prediciton = node.average_Q
                    y_predicted.append(prediciton)
                else:
                    raise ValueError('what type of tree leaves?')

        rmse_on_test_set = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_predicted))
        return rmse_on_test_set

    def print_tree_structure(self, print_mode, file_directory, means, variances, average_RMSE, number_train_data, number_test_data, rmse_on_test_set, running_time):
        root = self.root
        
        node_count = 0
        leaf_count = 0
        for i, node in self.nodes.items():
            node_count = node_count + 1
            if node.nodeType == NodeLeaf:
                leaf_count = leaf_count + 1

        with open(file_directory, 'w') as (f):
            header = ('"For each split, if condition is TRUE, go to the 1st node on the LEFT; \\nif condition is FALSE, go to the 2nd node on the LEFT; \\nif there are instances with split feature as PADDING 0s instead of values, there is a 3rd node for these instances. \\n \\n total_nodes: {} \\n total_leaf_nodes: {} \\n training_epochs: {} \\n min_number_of_instances_to_split: {} \\n split_method: {} \\n Q_value_difference_significant_level: {} \\n min_Q_difference_threshold: {} \\n number_of_gaps: {} \\n tree_max_depth: {} \\n pruning_method: {} \\n variance_reduction_prune: {} \\n pruning_lamb: {} \\n model_type: {} " \n\n').format(node_count, leaf_count, self.training_epochs, self.minSplitInstances, self.split_method, self.Q_value_difference_significant_level, self.min_Q_difference_threshold, self.number_of_gaps, self.max_depth, self.pruning_method, format(self.variance_reduction_prune, '.10f') if self.variance_reduction_prune is not None else None, format(self.pruning_lamb, '.10f') if self.pruning_lamb is not None else None, self.model_type)

            tree_structure = self.recursive_print_tree_structure(root, 0, means, variances, print_mode)

            conclusion = ('"average RMSE on all leaf nodes: {} \\n training data size: {} \\n test data size: {} \\n RMSE on test set: {} \\n Running time in seconds: {}"\n\n').format(average_RMSE, number_train_data, number_test_data, rmse_on_test_set if rmse_on_test_set is not None else None, running_time)

            print >> f, 'digraph {\n' + header + '\n' + conclusion + '\n' + tree_structure + '\n\n}'

    def recursive_print_tree_structure(self, node, layer, means, variances, print_mode):
        tree_structure = ''

        if node.idx == 1:
            node_graph = ''
        else:
            node_graph = ('{} -> {}').format(node.parent.idx, node.idx)

        if print_mode == 1: # print for Q
            average_type = 'Q'
        elif print_mode == 2: # print for Impact
            average_type = 'Impact'
        else:
            raise ValueError('Q or Impact?')

        if node.nodeType == NodeSplit:
            normalized_value = node.distinction.continuous_divide_value if node.distinction.continuous_divide_value is not None else node.distinction.discrete_divide_value

            # denormalize split value when printing tree, so it's easy to interpret

            # action is not normalized
            if self.problem.__class__.__name__ == 'SPG_ice_hockey':
                # only event features (index 0 to 11) are normalized
                normalization_index_bound = 12

                # 39 = (12 features + 27 one-hot action)
                normalization_index = node.distinction.dimension_index % 39

            elif self.problem.__class__.__name__ == 'SPG_soccer':
                # only some event features (index 0 to 15) are normalized
                normalization_index_bound = 16

                # 61 = (18 features + 43 one-hot action)
                normalization_index = node.distinction.dimension_index % 61

            if normalization_index < normalization_index_bound:  
                # too bad, the normalization process for ice hockey and soccer are different
                if self.problem.__class__.__name__ == 'SPG_ice_hockey':
                    mean = means[normalization_index]
                    variance = variances[normalization_index]
                    sd = math.sqrt(variance)
                    # our code uses standard deviation instead of variance for normalization
                    denormalizred_value = normalized_value * sd + mean

                elif self.problem.__class__.__name__ == 'SPG_soccer':
                    mean = means[normalization_index]
                    variance = variances[normalization_index]
                    denormalizred_value = normalized_value * variance + mean
            else:
                denormalizred_value = normalized_value
            
            if self.split_method == 4:
                split_criterion_type = 't-score'
            else:
                split_criterion_type = 'Variance Reduction'

            tree_structure += ('{} [label="Node ID: {} \\n Split Condition: {} {} {} ? \\n ' + split_criterion_type + ': {} \\n lambda: {} \\n Average {}: {} \\n Node Level: {}"] \n{}').format(
                node.idx,
                node.idx,
                node.distinction.dimension_name,
                '<=' if node.distinction.continuous_divide_value is not None else '==',
                format(denormalizred_value, '.5f'),
                format(node.distinction.split_criterion_value, '.10f'),
                node.lamb,
                average_type,
                node.qValues[0], # index 0 because only 1 action
                node.depth,
                node_graph
            )
            child_string = ''
            for child in node.children:
                child_string += '\n' + self.recursive_print_tree_structure(child, layer + 1, means, variances, print_mode)

            tree_structure += child_string
        else:
            if node.nodeType == NodeLeaf:
                tree_structure += '{} [label="Node ID: {} \\n Average {}: {} \\n RMSE: {} \\n lambda: {} \\n Number of Instances: {} \\n Node Level: {}"] \n{}'.format(
                    node.idx,
                    node.idx,
                    average_type,
                    node.qValues[0], # index 0 because only 1 action
                    format(node.rmse, '.10f') if node.rmse is not None else None,
                    node.lamb,
                    len(node.instances),
                    node.depth,
                    node_graph
                )
            else:
                raise ValueError(('Unsupported tree nodeType:{0}').format(node.nodeType))
        return tree_structure

    def getTime(self):
        """
        :return: length of history
        """
        return len(self.instances)

    def updateRoot(self, instance):
        """
        add the new instance ot LeafNode
        :param instance: instance to add
        :return:
        """
        self.insertInstance(instance)
        self.root.addInstance(instance)

        self.root.updateModel(action=instance.action, qValue=instance.qValue)

    def insertInstance(self, instance):
        """
        append new instance to history
        :param instance: current instance
        :return:
        """
        self.instances.append(instance)

    def modelFromInstances(self, node):
        """
        rebuild model for leaf node, with newly added instance
        :param node:
        :return:
        """
        node.count = np.zeros(self.n_actions)
        node.transitions = [{} for i in range(self.n_actions)]
        for inst in node.instances:
            node.updateModel(inst.action, inst.qValue)

    def getInstanceLeaf(self, inst, ntype=NodeLeaf, node=None):
        """
        Get leaf that inst records a transition from
        previous=0 indicates transition_from, previous=1 indicates transition_to
        :param inst: target instance
        :param ntype: target node type
        :param previous: previous=0 indicates present inst, previous=1 indicates next inst
        :return:
        """
        idx = inst.timestep

        if node is None:
            node = self.root

        while node.nodeType != ntype:
            child = node.applyDistinctionToInstance(inst)
            node = node.children[child]

        return node

    def genId(self):
        """
        :return: a new ID for node
        """
        self.node_id_count += 1
        return self.node_id_count

    def split(self, node, distinction):
        """
        split decision tree on a node
        :param node: node to split
        :param distinction: distinction to split
        :return:
        """
        node.nodeType = NodeSplit
        node.distinction = distinction
        
        if distinction.has_paddings == True:
            number_of_nodes = 3 # one more node for instances having paddings on the split feature
        else:
            number_of_nodes = 2 # only consider binary tree otherwise

        for i in range(number_of_nodes): 
            idx = self.genId() # ID/Index will be assigned to the new child Node
            child_node = Tree_Node(idx, NodeLeaf, node, self.n_actions, node.depth + 1, self.problem)
            self.nodes[idx] = child_node
            node.children.append(child_node)
            child_node.weight = node.weight
            child_node.bias = node.bias

        # transfer the instances on current node to child nodes
        print >> sys.stderr, 'len(node.instances): {}'.format(len(node.instances))
        for instance in node.instances:
            child_node_for_instance = self.getInstanceLeaf(instance, node=node)
            child_node_for_instance.addInstance(instance)

        # print the number of instances transferred to each child node
        for child_node in node.children:
            print >> sys.stderr, 'len(child_node_{}.instances): {}'.format(child_node.idx, len(child_node.instances))

        for i, n in self.nodes.items():
            if n.nodeType == NodeLeaf:
                self.modelFromInstances(n)

        node.instances = []

    def pick_model_and_train(self, node):
        # self.train_linear_regression_on_leaves(node)
        if self.model_type == 0:
            raise ValueError(('null model should not reach here:{}').format(self.model_type))
        if self.model_type == 1:
            self.calculate_linear_regression_on_leaves(node)
        elif self.model_type == 2:
            self.calculate_average_Q_on_leaves(node)
        else:
            raise ValueError(('Unsupported model_type:{}').format(self.model_type))

    def testFringe(self):
        """
        Tests fringe nodes for viable splits, splits nodes if they're found
        :return: how many real splits it takes
        """
        return self.testFringeRecursive(self.root)

    def testFringeRecursive(self, node):
        """
        recursively perform test in fringe, until return total number of split
        :param node: node to test
        :return: number of splits
        """
        if self.max_depth is not None and node.depth >= self.max_depth:
            if node.nodeType == NodeLeaf:
                # in order to still get node.average_diff when max_depth is reached
                self.pick_model_and_train(node)

            print('max_depth ', self.max_depth, ' reached.')
            return 0

        if node.nodeType == NodeLeaf:
            start = time.time()
            
            self.pick_model_and_train(node)

            end = time.time()
            print >> sys.stderr, '--- time used for training linear model on current node {}: {}'.format(node.idx, end - start)

            BestD = self.getUtileDistinction(node)

            if BestD:
                self.split(node, BestD)

                return 1 + self.testFringeRecursive(node)
            return 0

        total = 0
        for c in node.children:
            total += self.testFringeRecursive(c)

        return total

    def getUtileDistinction(self, node):
        """
        Different kinds of tests are performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform test until find the proper distinction, otherwise, return None
        """
        if len(node.instances) < self.minSplitInstances:
            print('number of instances ', len(node.instances), ' on the node is less than minSplitInstances ', self.minSplitInstances)
            return None
        
        start = time.time()

        candidate_distinctions = self.getCandidateDistinctions(node)

        end = time.time()
        print >> sys.stderr, '--- time used for finding all candidate distinctions: {}'.format(end - start)
        
        print >> sys.stderr, 'total number of Candidate Distinctions: {}'.format(len(candidate_distinctions))

        return self.ksTestonQ(node, candidate_distinctions)

    def splitQs(self, node, candidate_distinction):

        event_number = 0
        if candidate_distinction.dimension_name != 'actions':
            event_number = int(candidate_distinction.dimension_name[-1])

        if candidate_distinction.iscontinuous:
            Q_value_list = []
            for i in range(0, 2):
                Q_value_list.append([])
            
            # if padding is on the event is possible, add one more list for instances with no values but paddings
            # event 0 has no paddings
            if event_number >= 1:
                Q_value_list.append([])

            for inst in node.instances:

                if event_number >= 1:
                    # if the instance has no values on the split candidate feature but paddings instead, then add it to the 3rd list which represents all instances that have no value on the split feature
                    event_is_padding = Util.isEventPaddings(event_number, inst, self.problem)
                    if event_is_padding == True:
                        Q_value_list[2].append(inst.qValue)
                        continue

                if inst.currentObs[candidate_distinction.dimension_index] <= candidate_distinction.continuous_divide_value:
                    Q_value_list[0].append(inst.qValue)
                else:
                    Q_value_list[1].append(inst.qValue)

        else:
            Q_value_list = []

            if candidate_distinction.dimension_name is 'actions':
                # because all data are collected based on only 1 action
                dimension = 1
            else:
                dimension = 2 # we only consider binary tree

            for i in range(0, dimension):
                    Q_value_list.append([])

            # if padding is on the event is possible, add one more list for instances with no values but paddings
            # event 0 has no paddings
            if event_number >= 1:
                Q_value_list.append([])
            
            for inst in node.instances:
                if event_number >= 1:
                    # if the instance has no values on the split candidate feature but paddings instead, then add it to the 3rd list which represents all instances that have no value on the split feature
                    event_is_padding = Util.isEventPaddings(event_number, inst, self.problem)
                    if event_is_padding == True:
                        Q_value_list[2].append(inst.qValue)
                        continue
                    
                if candidate_distinction.dimension_name is 'actions' or inst.currentObs[candidate_distinction.dimension_index] == candidate_distinction.discrete_divide_value:
                    Q_value_list[0].append(inst.qValue)
                else:
                    Q_value_list[1].append(inst.qValue)

        return Q_value_list

    def ksTestonQ(self, node, candidate_distinctions):
        """
        KS test is performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform ks test until find the proper distinction, otherwise, return None
        """
        assert node.nodeType == NodeLeaf

        root_utils = self.getQs(node)
        variance = np.var(root_utils)

        best_distinction_to_split = None
        best_split_criterion_value = 0

        start = time.time()

        for candidate_distinction in candidate_distinctions:
            # split the node based on the candidate distinction
            Qs_on_fringe_nodes = self.splitQs(node, candidate_distinction)

            split_criterion_value = 0
            not_enough_instances_on_child_node = False

            # skip the invalid split on action
            if len(Qs_on_fringe_nodes) < 2:
                continue

            if self.split_method == 4:
                # Splitting Criterion: t-test
                a = Qs_on_fringe_nodes[0]
                b = Qs_on_fringe_nodes[1]
                if len(a) < self.minSplitInstances or len(b) < self.minSplitInstances:
                    not_enough_instances_on_child_node = True
                else:
                    # use Welch's t-test because sample variances and sample sizes of a and b are different
                    t, p = ttest_ind(a, b, equal_var=False)
                    split_criterion_value = t

            else:
                # Splitting Criterion: variance reduction
                for i, Qs_on_fringe_node in enumerate(Qs_on_fringe_nodes):
                    if i == 2:
                        if len(Qs_on_fringe_node) == 0:
                            continue
                        else:
                            candidate_distinction.has_paddings = True

                    if len(Qs_on_fringe_node) == 0:
                        not_enough_instances_on_child_node = True
                        break
                    else:
                        number_of_instances_child_node = len(Qs_on_fringe_node)
                        number_of_instances_current_node = len(node.instances)

                        # do not split if any resulting child node has too less instances
                        # it's okay if the list of instances with paddings has too less instances
                        if number_of_instances_child_node < self.minSplitInstances and i < 2:
                            not_enough_instances_on_child_node = True
                            break

                        
                        weight = float(number_of_instances_child_node) / float(number_of_instances_current_node)

                        variance_child = np.var(Qs_on_fringe_node)
                        variance_reduction = variance - variance_child 
                        total_variance_reduction = split_criterion_value + weight * variance_reduction
                        split_criterion_value = total_variance_reduction

            if not_enough_instances_on_child_node:
                continue
            
            if split_criterion_value > best_split_criterion_value:
                best_split_criterion_value = split_criterion_value
                candidate_distinction.split_criterion_value = split_criterion_value
                best_distinction_to_split = candidate_distinction

        end = time.time()
        print >> sys.stderr, '--- time used for finding best distinction to split: {}'.format(end - start)

        if best_distinction_to_split:
            print >> sys.stderr, 'Will split on distinction {}'.format(best_distinction_to_split.dimension_name)
            return best_distinction_to_split
        else:
            print >> sys.stderr, 'No good split.'

        print >> sys.stderr, 'best_split_criterion_value: {}'.format(best_split_criterion_value)
        return best_distinction_to_split

    def getQs(self, node):
        """
        Get all expected future discounted returns for all instances in a node
        (q-value is just the average EFDRs)
        """
        efdrs = np.zeros(len(node.instances))
        for i, inst in enumerate(node.instances):
            efdrs[i] = inst.qValue
        return [efdrs]

    # take out 1/10 data for testing
    def split_train_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=666)
        return x_train, x_test, y_train, y_test
        
    def calculate_linear_regression_on_leaves(self, node):
        x = []
        y = []
        for instance in node.instances:
            # add one more value on x as bias term
            x.append(instance.currentObs + [1.0])
            # x.append(instance.currentObs)
            y.append(instance.qValue)

        x = np.array(x)
        y = np.array(y)

        # use hold-out test set to calculate RMSE 
        x_train, x_test, y_train, y_test = self.split_train_test(x, y)

        # use pseudo inverse to calculate weights
        # pseudo_inverse = np.linalg.pinv(x_train)
        # w = pseudo_inverse.dot(y_train)
        # node.weight = w

        # since weights are huge, so add regularization
        # try all possible lambda values and choose the best one
        best_rmse = None
        best_weight = None
        best_lambda = None
        lambda_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for lamb in lambda_values:
            n_col = x_train.shape[1]
            w = np.linalg.lstsq(x_train.T.dot(x_train) + lamb * np.identity(n_col), x_train.T.dot(y_train))
            weights = w[0]

            # calculate RMSE
            y_predicted = x_test.dot(weights.T)

            rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_predicted))

            if best_rmse is None or rmse <= best_rmse:
                best_weight = weights
                best_lambda = lamb
                best_rmse = rmse

        node.weight = best_weight
        node.lamb = best_lambda
        node.rmse = best_rmse
    
    def calculate_average_Q_on_leaves(self, node):
        Qs = []
        for instance in node.instances:
            Qs.append(instance.qValue)

        average_Q = sum(Qs)/float(len(Qs))

        node.average_Q = average_Q

        average_Q_list = []
        for i in range(0, len(Qs)):
            average_Q_list.append(average_Q)

        rmse = math.sqrt(mean_squared_error(y_true=Qs, y_pred=average_Q_list))
        node.rmse = rmse
        
    def SegmentedLinearReg(self, X, Y, breakpoints ):
        ramp = lambda u: np.maximum( u, 0 )
        step = lambda u: ( u > 0 ).astype(float)

        breakpoints = np.sort( np.array(breakpoints) )

        dt = np.min( np.diff(X) )
        ones = np.ones_like(X)

        nIterationMax = 10
        for i in range( nIterationMax ):
            # Linear regression:  solve A*p = Y
            Rk = [ramp( X - xk ) for xk in breakpoints ]
            Sk = [step( X - xk ) for xk in breakpoints ]
            A = np.array([ ones, X ] + Rk + Sk )
            p =  lstsq(A.transpose(), Y, rcond=None)[0] 

            # Parameters identification:
            a, b = p[0:2]
            ck = p[ 2:2+len(breakpoints) ]
            dk = p[ 2+len(breakpoints): ]

            # Estimation of the next break-points:
            newBreakpoints = breakpoints - dk/ck 

            # Stop condition
            if np.max(np.abs(newBreakpoints - breakpoints)) < dt/5:
                break

            breakpoints = newBreakpoints

            # breakpoint is NaN means there is no optimal breakpoint
            if math.isnan(breakpoints[0]):
                return []

        else:
            pass

        return breakpoints

    def getCandidateDistinctions(self, node):
        """
        construct all candidate distinctions
        :param node: target nodes
        :return: all candidate distinctions
        """
        p = node.parent
        anc_distinctions = []
        while p:
            anc_distinctions.append(p.distinction)
            p = p.parent

        # Q-gap (min threshold)
        if self.split_method == 0:
            if self.Q_value_difference_significant_level is None:
                Q_difference_threshold = len(node.instances) / 4500000.000 # came up with this numbr by 265000/0.06, where 265000 is the number of training data in our training file, and 0.06 is the threshold I used for hardcoded Q_value_difference_significant_level

                # if the Q difference between 2 data is smaller than min_Q_difference_threshold, we assume their Q values are the same 
                if Q_difference_threshold < self.min_Q_difference_threshold:
                    Q_difference_threshold = self.min_Q_difference_threshold
            else:
                Q_difference_threshold = self.Q_value_difference_significant_level

        # get 2 Gaussian Mixture Clusters
        elif self.split_method == 1:
            x_list = []
            for instance in node.instances:
                x_list.append(instance.currentObs)
            
            x = np.array(x_list)

            # gmm = GaussianMixture(n_components=2, covariance_type='full').fit(x)
            gmm = GaussianMixture(n_components=2, covariance_type='diag').fit(x)        
            # prediction_gmm = gmm.predict(x)
            # probs = gmm.predict_proba(x)

        # segmented/piecewise regression
        elif self.split_method == 2:
            pass
        
        # calculate variance reduction incrementally
        # or, calucalte t-test incrementally
        elif self.split_method == 3 or self.split_method == 4:
            # calculate the varaince of the current node before split
            total_instance_number = len(node.instances)
            total_q = 0
            total_q_square = 0
            for instance in node.instances:
                total_q = total_q + instance.qValue
                total_q_square = total_q_square + (instance.qValue) ** 2
            
            total_mean = total_q / float(total_instance_number)
            variance = (total_q_square - 2 * total_mean * total_q + total_instance_number * (total_mean ** 2)) / total_instance_number
        
        # Q-gap (top) or Q-gap-gradient (top)
        elif self.split_method == 5 or self.split_method == 6:
            pass
        
        else:
            raise ValueError(('Unsupported split_method: {}').format(self.split_method))

        candidate_distinctions = []
        for i in range(self.max_back_depth):
            for j in range(-1, self.n_dim):
                
                if j > -1 and self.dim_sizes[j] == 'continuous':
                    # Q-gap (min threshold) or Q-gap (top) or # Q-gap-gradient (top)
                    if self.split_method == 0 or self.split_method == 5 or self.split_method == 6:

                        if self.split_method == 5 or self.split_method == 6:
                            # key: split_value
                            # value: distance_in_Q
                            top_splitvalue_gap = dict()

                        # sort data by each feature
                        sorted_instances = sorted(node.instances, key=lambda inst: inst.currentObs[j])
                        for i in range(0, len(sorted_instances) - 1):
                            # distance in feature
                            distance_in_feature = abs(sorted_instances[i+1].currentObs[j] - sorted_instances[i].currentObs[j])

                            # if distance in feature is 0 then ignore, because if 2 data have the same current feature value but Q value difference is high, then the difference in Q is not due to the current feature
                            if distance_in_feature == 0:
                                continue

                            # distance in Q
                            distance_in_Q = abs(sorted_instances[i+1].qValue - sorted_instances[i].qValue)

                            # Q-gap-gradient (top)
                            if self.split_method == 6:
                                # distance_in_feature is not 0
                                # calculate Q-gap gradient
                                distance_in_Q = distance_in_Q / float(distance_in_feature)

                            # Q-gap (top) or Q-gap-gradient (top)
                            if self.split_method == 5 or self.split_method == 6:
                                if len(top_splitvalue_gap.items()) > 0:
                                    # find the key with the smallest gap, then get the smallest gap
                                    key_min = min(top_splitvalue_gap.keys(), key=(lambda k: top_splitvalue_gap[k]))
                                    smallest_gap = top_splitvalue_gap[key_min]
                                else:
                                    smallest_gap = 0
                            # Q-gap (min threshold)
                            else:
                                smallest_gap = Q_difference_threshold

                            if distance_in_Q > smallest_gap:

                                # split in the middle of the feature
                                split_value = (sorted_instances[i+1].currentObs[j] + sorted_instances[i].currentObs[j]) / 2.0

                                # Q-gap (top) or Q-gap-gradient (top)
                                if self.split_method == 5 or self.split_method == 6:
                                    if len(top_splitvalue_gap.items()) >= self.number_of_gaps:
                                        del top_splitvalue_gap[key_min]
                                
                                    top_splitvalue_gap[split_value] = distance_in_Q
                                # Q-gap (min threshold)
                                else:
                                    d = Distinction(dimension_index=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True, continuous_divide_value = split_value)
                                    
                                    if d in anc_distinctions:
                                        continue
                                    else:
                                        candidate_distinctions.append(d)

                            # in order to solve XOR problem, if the current instance and the next instance are very similar in Q, and if the next instance and next next instance have the same feature value, compare the Q values of the current instance and the next next intance. Repeat, until a new instance with a different feature value
                            else:
                                offset = 0
                                while True:
                                    offset = offset + 1

                                    if (i + offset + 1) >= len(sorted_instances):
                                        break

                                    # if the next instance and next next instance have the same feature value
                                    if sorted_instances[i + offset].currentObs[j] == sorted_instances[i + offset + 1].currentObs[j]:
                                        # distance in Q of the current instance and the next next instance
                                        distance_in_Q = abs(sorted_instances[i + offset + 1].qValue - sorted_instances[i].qValue)

                                        # Q-gap-gradient (top)
                                        if self.split_method == 6:
                                            # distance_in_feature is not 0 and is the same as before
                                            # calculate Q-gap gradient
                                            distance_in_Q = distance_in_Q / float(distance_in_feature)
                                        
                                        if distance_in_Q > smallest_gap:
                                            # split in the middle of the feature
                                            split_value = (sorted_instances[i + offset + 1].currentObs[j] + sorted_instances[i].currentObs[j]) / 2.0

                                            # Q-gap (top) or Q-gap-gradient (top)
                                            if self.split_method == 5 or self.split_method == 6:
                                                if len(top_splitvalue_gap.items()) >= self.number_of_gaps:
                                                    del top_splitvalue_gap[key_min]
                                                
                                                top_splitvalue_gap[split_value] = distance_in_Q
                                            # Q-gap (min threshold)
                                            else:
                                                d = Distinction(dimension_index=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True, continuous_divide_value = split_value)

                                                if d in anc_distinctions:
                                                    continue
                                                else:
                                                    candidate_distinctions.append(d)

                                            break
                                    else:
                                        break
                        
                        # Q-gap (top) or Q-gap-gradient (top)
                        if self.split_method == 5 or self.split_method == 6:
                            if len(top_splitvalue_gap.items()) > 0:
                                for split_value in top_splitvalue_gap.keys():
                                    d = Distinction(dimension_index=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True, continuous_divide_value = split_value)

                                    if d in anc_distinctions:
                                        continue
                                    else:
                                        candidate_distinctions.append(d)

                    # for each continuous feature, solve equation to find the split point that best separates the 2 Gaussian Mixture Clusters
                    elif self.split_method == 1:
                        weight1 = gmm.weights_[0]
                        weight2 = gmm.weights_[1]
                        mean1 = gmm.means_[0][j]
                        mean2 = gmm.means_[1][j]
                        varaince1 = gmm.covariances_[0][j]
                        varaince2 = gmm.covariances_[1][j]

                        x = Symbol('x')

                        # original equation but very slow to solve
                        # equation = weight1 * (1.0/(math.sqrt(varaince1) * math.sqrt(2.0 * math.pi))) * math.e**(-(mean1-x)**2 / (2.0*varaince1)) - weight2 * (1.0/(math.sqrt(varaince2) * math.sqrt(2.0 * math.pi))) * math.e**(-(mean2-x)**2 / (2.0*varaince2))

                        # if the 2 variances are close, preferable to solve this equation
                        equation = 2.0 * x * (mean1-mean2) - mean1**2 + mean2**2 + 2.0 * varaince1**2 * math.log(weight1/weight2)

                        ans = solve(equation, x)

                        if len(ans) > 0:
                            d = Distinction(dimension_index=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True, continuous_divide_value = ans[0])
                                
                            if d in anc_distinctions:
                                continue
                            else:
                                candidate_distinctions.append(d)

                    # find the segmented regression on each feature
                    elif self.split_method == 2:
                        x_list = []
                        y_list = []
                        for instance in node.instances:
                            x_list.append(instance.currentObs[j])
                            y_list.append(instance.qValue)
                        
                        # use (max_X + min_X) / 2 as the initialBreakpoint
                        min_X = min(x_list)
                        max_X = max(x_list)
                        initialBreakpoint_X = [(min_X + max_X) / 2.0]

                        breakpoint_X = self.SegmentedLinearReg(x_list, y_list, initialBreakpoint_X)

                        if len(breakpoint_X) > 0 and not math.isnan(breakpoint_X[0]):
                            d = Distinction(dimension_index=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True, continuous_divide_value = breakpoint_X[0])

                            if d in anc_distinctions:
                                continue
                            else:
                                candidate_distinctions.append(d)
                    
                    # calculate variance reduction incrementally
                    # or, calucalte t-test incrementally
                    # TODO : for variance calculation, divide by N-1 instead of N
                    elif self.split_method == 3 or self.split_method == 4:
                        # take out paddings before sorting so that the calculation on the left side doesn't contain paddings
                        event_number = int(self.dim_names[j][-1])
                        # event 0 has no paddings
                        if event_number >= 1:
                            padding_instances = []
                            instances_without_paddings = []
                            for instance in node.instances:
                                event_is_padding = Util.isEventPaddings(event_number, instance, self.problem)
                                if event_is_padding == True:
                                    padding_instances.append(instance)
                                else:
                                    instances_without_paddings.append(instance)
                        else:
                            padding_instances = []
                            instances_without_paddings = node.instances

                        # sort data (without paddings) by each feature
                        sorted_instances = sorted(instances_without_paddings, key=lambda inst: inst.currentObs[j])

                        # put paddings at the end for variance reduction calculation
                        if self.split_method == 3 or self.split_method == 4:
                            sorted_instances = sorted_instances + padding_instances

                        # find the best variance reduction incrementally by going though each data
                        best_split_criterion_value = 0
                        split_value = None

                        left_q = 0
                        left_q_square = 0
                        right_q = total_q
                        right_q_square = total_q_square
                        left_instance_number = 0
                        for i in range(0, len(sorted_instances) - 1):
                            # incrementally update left side
                            left_instance_number = left_instance_number + 1.0
                            left_q = left_q + sorted_instances[i].qValue
                            left_q_square = left_q_square + (sorted_instances[i].qValue) ** 2
                            left_mean = left_q / float(left_instance_number)

                            # incrementally update right side
                            right_instance_number = total_instance_number - left_instance_number
                            right_q = right_q - sorted_instances[i].qValue
                            right_q_square = right_q_square - (sorted_instances[i].qValue) ** 2
                            right_mean = right_q / float(right_instance_number)

                            # do not consider if any resulting child node has too less instances
                            if left_instance_number < self.minSplitInstances or right_instance_number < self.minSplitInstances:
                                continue

                            left_variance = (left_q_square - 2 * left_mean * left_q + left_instance_number * (left_mean ** 2)) / float(left_instance_number)

                            right_variance = (right_q_square - 2 * right_mean * right_q + right_instance_number * (right_mean ** 2)) / float(right_instance_number)

                            if self.split_method == 3:
                                variance_reduction = variance - (left_instance_number / float(total_instance_number)) * left_variance - (right_instance_number / float(total_instance_number)) * right_variance

                                split_criterion_value = variance_reduction

                            elif self.split_method == 4:
                                # use Welch's t-test because sample variances and sample sizes between 2 groups are different
                                sqrt_value = math.sqrt(left_variance / float(left_instance_number) + right_variance / float(right_instance_number))

                                if sqrt_value == 0:
                                    sqrt_value = 0.0000000001

                                t = float(left_mean - right_mean) / sqrt_value
                            
                                split_criterion_value = t

                            if split_criterion_value > best_split_criterion_value:
                                best_split_criterion_value = split_criterion_value
                                # split in the middle of the feature
                                split_value = (sorted_instances[i+1].currentObs[j] + sorted_instances[i].currentObs[j]) / 2.0

                        if split_value is not None:
                            d = Distinction(dimension_index=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True, continuous_divide_value = split_value)

                            if d in anc_distinctions:
                                continue
                            else:
                                candidate_distinctions.append(d)

                    else:
                        raise ValueError(('Unsupported split_method: {}').format(self.split_method))

                else:
                    if j == -1:
                        d = Distinction(dimension_index = j, back_idx = i, dimension_name = 'actions')

                        if d in anc_distinctions:
                            continue
                        else:
                            candidate_distinctions.append(d)
                    else:
                        possible_discrete_values = set()
                        for inst in node.instances:
                            current_discrete_value = inst.currentObs[j]
                            if current_discrete_value not in possible_discrete_values:
                                d = Distinction(dimension_index = j, back_idx = i, dimension_name = self.dim_names[j] if j > -1 else 'actions', discrete_divide_value = current_discrete_value)

                                possible_discrete_values.add(current_discrete_value)
                    
                                if d in anc_distinctions:
                                    continue
                                else:
                                    candidate_distinctions.append(d)

        return candidate_distinctions


class Tree_Node:
    def __init__(self, idx, nodeType, parent, n_actions, depth, problem):
        self.idx = idx
        self.nodeType = nodeType
        self.parent = parent
        self.children = []
        self.count = np.zeros(n_actions)
        self.qValues = np.zeros(n_actions)
        self.distinction = None
        self.instances = []
        self.depth = depth
        self.weight = None
        self.bias = None
        self.average_diff = None
        self.update_times = 0
        self.lamb = None
        self.rmse = None
        self.problem = problem
        self.average_Q = None

    def addInstance(self, instance):
        """
        add new instance to node instance list
        if instance length exceed maximum history length, select most recent history
        :param instance:
        :return:
        """
        self.instances.append(instance)

    def updateModel(self, action, qValue):
        """
        1. add action reward
        2. add action count
        3. record transition states
        :param new_state: new transition state
        :param action: new action
        :param reward: reward of action
        :param home_identifier: identify home and away
        :return:
        """
        self.qValues[action] = float((self.qValues[action] * self.count[action] +
                                qValue)) / float((self.count[action] + 1))

        self.count[action] += 1

    def applyDistinctionToInstance(self, inst):
        if self.distinction.dimension_index == ActionDimension:
            return inst.action

        if self.distinction.has_paddings == True:
                event_number = int(self.distinction.dimension_name[-1])
                # event 0 has no paddings
                if event_number >= 1:
                    event_is_padding = Util.isEventPaddings(event_number, inst, self.problem)
                    if event_is_padding == True:
                        return 2

        if self.distinction.iscontinuous:
            if inst.currentObs[self.distinction.dimension_index] <= self.distinction.continuous_divide_value:
                return 0
            else:
                return 1
        else:
            if inst.currentObs[self.distinction.dimension_index] == self.distinction.discrete_divide_value:
                return 0
            else:
                return 1

class Instance:
    """
    records the transition as an instance
    """

    def __init__(self, timestep, state_features, action, qValue):
        self.timestep = int(timestep)
        self.action = int(action)

        next_features = copy.deepcopy(state_features)
        for i in range(0, len(next_features)):
            next_features[i] = -1 # make all features of next state -1
        
        self.nextObs = map(float, next_features) 

        self.currentObs = map(float, state_features)
        self.qValue = qValue
        self.state_features = state_features


class Distinction:
    """
    For split node
    """

    def __init__(self, dimension_index, back_idx, dimension_name='unknown', iscontinuous=False, continuous_divide_value=None, discrete_divide_value=None):
        """
        initialize distinction
        :param dimension_index: split of the node is based on the dimension
        :param back_idx: history index, how many time steps backward from the current time this feature will be examined
        :param dimension_name: the name of dimension
        :param iscontinuous: continuous or not
        :param continuous_divide_value: the value of continuous division
        """
        self.dimension_index = dimension_index
        self.back_idx = back_idx
        self.dimension_name = dimension_name
        self.iscontinuous = iscontinuous
        self.continuous_divide_value = continuous_divide_value
        self.discrete_divide_value = discrete_divide_value
        self.split_criterion_value = None
        self.has_paddings=False

    def __eq__(self, distinction):
        return self.dimension_index == distinction.dimension_index and self.back_idx == distinction.back_idx and self.continuous_divide_value == distinction.continuous_divide_value
