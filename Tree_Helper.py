import gc
import numpy as np, ast, scipy.io as sio, os, unicodedata, pickle, csv
import Tree_Structures
from scipy.stats import pearsonr
import time
import math
from sklearn.metrics import mean_squared_error
import sys
import os
import random

class Helper:
    def __init__(self, save_directory, training_file_name, problem, minSplitInstances, Q_value_difference_significant_level, min_Q_difference_threshold, training_epochs, max_depth, variance_reduction_prune, number_of_gaps, split_method, model_type, pruning_method, pruning_lamb, held_out_test_set, training_mode=''):
        self.utree = Tree_Structures.Model_Tree(gamma=problem.gamma, n_actions=len(problem.actions), dim_sizes=problem.dimSizes, dim_names=problem.dimNames, problem = problem, training_mode=training_mode, minSplitInstances = minSplitInstances, Q_value_difference_significant_level=Q_value_difference_significant_level, min_Q_difference_threshold=min_Q_difference_threshold, training_epochs = training_epochs, max_depth=max_depth, variance_reduction_prune=variance_reduction_prune, number_of_gaps=number_of_gaps, split_method=split_method, pruning_method=pruning_method, pruning_lamb=pruning_lamb, model_type=model_type)
      
        self.valiter = 1
        self.problem = problem

        self.SAVE_PATH = save_directory + '/split_method_{}/'.format(split_method)
        
        if not os.path.isdir(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)

        self.PRINT_TREE_PATH = self.SAVE_PATH + '/learned_tree{}_{}_{}.txt'.format(training_mode, training_file_name, time.time())
        self.training_mode = training_mode

        self.variance_reduction_prune = variance_reduction_prune

        self.model_type = model_type

        self.pruning_method = pruning_method

        self.held_out_test_set = held_out_test_set

    def update(self, state_features, action, qValue, value_iter=0, check_fringe=0, home_identifier=None):
        t = self.utree.getTime()
        data_instance = Tree_Structures.Instance(t, state_features, action, qValue)
        self.utree.updateRoot(data_instance)
        if check_fringe:
            self.utree.testFringe()

    def read_SPG_csv_data(self, csvfile):
        Qs = list()
        states = list()
        hasHeader = True

        with open(csvfile, 'r') as fp:
            csv_file_iterator = csv.reader(fp, delimiter=',')
            
            for row in csv_file_iterator:

                # skip header
                if hasHeader == True:
                    hasHeader = False
                    continue

                row_float = list()
                isQ = True
                for element in row:
                    if isQ == True:
                        Qs.append(float(element))
                        isQ = False
                        continue

                    row_float.append(float(element))
                
                states.append(row_float)
            
            states = np.array(states)
            Qs = np.array(Qs)
            print >> sys.stderr, 'training_data length: {}'.format(len(states))
            fp.close()
        return states, Qs     

    def read_Guide_concrete_csv_data(self, csvfile):
        Qs = list()
        states = list()
        hasHeader = False

        with open(csvfile, 'r') as fp:
            csv_file_iterator = csv.reader(fp, delimiter=',')
            
            for row in csv_file_iterator:

                # skip header
                if hasHeader == True:
                    hasHeader = False
                    continue

                row_float = list()
                # isQ = True
                count = 0
                for element in row:

                    count = count + 1

                    # ignore row number, Slump data and Flow data
                    if count == 1 or count == 9 or count == 10:
                        continue
                    
                    # this is the training Y data
                    if count == 11:
                        Qs.append(float(element))
                        continue
                    
                    row_float.append(float(element))
                
                states.append(row_float)
            
            states = np.array(states)
            Qs = np.array(Qs)
            print('training_data length: ', len(states))
            fp.close()
        return states, Qs    

    def read_Guide_derm_csv_data(self, csvfile):
        Qs = list()
        states = list()

        with open(csvfile, 'r') as fp:
            csv_file_iterator = csv.reader(fp, delimiter=',')
            
            for row in csv_file_iterator:

                row_float = list()
                # isQ = True
                count = 0
                for element in row:

                    count = count + 1
                    
                    # this is the training Y data
                    if count == 35:
                        Qs.append(float(element))
                        continue
                    
                    row_float.append(float(element))
                
                states.append(row_float)
            
            states = np.array(states)
            Qs = np.array(Qs)
            print('training_data length: ', len(states))
            fp.close()
        return states, Qs  

    def read_Guide_birthwt_csv_data(self, csvfile):
        Qs = list()
        states = list()

        with open(csvfile, 'r') as fp:
            csv_file_iterator = csv.reader(fp, delimiter=',')
            
            for row in csv_file_iterator:

                row_float = list()
                count = 0
                for element in row:

                    count = count + 1

                    # this is the training Y data
                    if count == 1:
                        Qs.append(float(element))
                        continue

                    # ignore lowbwt
                    if count == 11:
                        continue
                    
                    row_float.append(float(element))
                
                states.append(row_float)
            
            states = np.array(states)
            Qs = np.array(Qs)
            print('training_data length: ', len(states))
            fp.close()
        return states, Qs    

    def read_number_by_number(self, file):
        values = []
        with open(file, 'r') as f:
            for line in f:
                for value_str in line.split(' '):
                    value_str_stripped = value_str.strip().strip('[]')
                    if len(value_str_stripped) != 0:
                        values.append(float(value_str_stripped))
        
        return values

    def read_mean_variance_texts(self, normalization_file_directory):
        mean_file = normalization_file_directory + 'feature_mean.txt'
        means = self.read_number_by_number(mean_file)
        # print(means)

        variance_file = normalization_file_directory + 'feature_var.txt'
        variances = self.read_number_by_number(variance_file)
        # print(variances)

        return means, variances

    def episode(self, print_mode, file_name, normalization_file_directory, start_time, timeout=int(100000.0), save_checkpoint_flag=0):
        count = 0

        states, Qs = self.read_SPG_csv_data(file_name)

        # states, Qs = self.read_Guide_concrete_csv_data(file_name)

        # states, Qs = self.read_Guide_derm_csv_data(file_name)

        # states, Qs = self.read_Guide_birthwt_csv_data(file_name)

        count += 1
        
        if self.held_out_test_set == True:
            # use held-out test set 
            x_train, x_test, y_train, y_test = self.utree.split_train_test(states, Qs)
        else:
            # use whole dataset as test set 
            x_train = states
            y_train = Qs
            x_test = states
            y_test = Qs

        # calculate RMSE with null_model
        if self.model_type == 0:
            average_Q = sum(y_train)/float(len(y_train))
            print >> sys.stderr, 'average_Q: {}'.format(average_Q)

            average_Q_list = []
            for i in range(0, len(y_test)):
                average_Q_list.append(average_Q)

            rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=average_Q_list))
            print >> sys.stderr, 'RMSE: {}'.format(rmse)
            return

        event_number = len(x_train)

        for index in range(0, event_number):
            # print ('data: ', index, '/', event_number)

            if self.problem.isEpisodic:
                state_features = x_train[index]
                qValue = y_train[index]
                # all training data are collected based on the same action, therefore, use hard coded action
                action = 0 

                if index == event_number - 1:  # game ending
                    print >> sys.stderr, '=============== update starts ==============='  
                    self.update(state_features, action, qValue, value_iter=1, check_fringe=1)
                    print >> sys.stderr, '=============== update finished ===============\n'
                else:
                    self.update(state_features, action, qValue, check_fringe=0)
        
        if self.pruning_method == 1: # pruning by variance reduction 
            # prune intermediate nodes to leaf nodes if neither the node itself nor any of its descendant nodes have a split with good variance reduction
            for i, node in self.utree.nodes.items():
                good_variance_reduction = self.utree.descendantHasGoodVarianceReduction(self.variance_reduction_prune, node)
                # print('Node ID: ', node.idx, ' has Good Variance Reduction: ', good_variance_reduction)

                if good_variance_reduction == False:
                    self.utree.prune_intermediate_node(node)
        
        elif self.pruning_method == 2 or self.pruning_method == 3: # pruning by squared error
            # check for any leaf node if SE(parent) <= SE(leaf1) + SE(leaf2) + SE(leaf3), if yes, prune
            while True:
                finished = self.utree.prune_leaf_nodes()
                if finished == True:
                    break

        # for printing tree with denormalized values
        means, variances = self.read_mean_variance_texts(normalization_file_directory)

        average_RMSE = self.utree.calculate_average_RMSE(total_instances_number=event_number)

        end_time = time.time()
        running_time = end_time - start_time

        rmse_on_test_set = self.utree.RMSE_on_test_set(x_test, y_test)

        print("running_time: {} seconds".format(running_time))

        print '*** Writing Game File ***\n'
        self.utree.print_tree_structure(print_mode, self.PRINT_TREE_PATH, means, variances, average_RMSE, len(x_train), len(x_test), rmse_on_test_set, running_time)
        print '*** Game File Done ***\n'
