import Problem_SPG_ice_hockey
import Problem_SPG_soccer
import Tree_Helper
import os
import itertools
import time
import sys

if __name__ == "__main__":
    training_file_path = ''
    save_directory = ''
    normalization_file_directory = ''

    training_file_names = []

    # problem = Problem_SPG_ice_hockey.SPG_ice_hockey()
    problem = Problem_SPG_soccer.SPG_soccer()

    if problem.__class__.__name__ == 'SPG_ice_hockey':
        training_file_path = training_file_path + '/ice_hockey/'
        normalization_file_directory = normalization_file_directory + '/ice_hockey/'

        training_file_names.append('Q_shot_2019_11_13_11_14_23')
        training_file_names.append('impact_shot_2019_11_13_11_14_23')
    
        training_file_names.append('Q_pass_2019_11_19_18_12_33')
        training_file_names.append('impact_pass_2019_11_19_18_12_33')

    elif problem.__class__.__name__ == 'SPG_soccer':
        training_file_path = training_file_path + '/soccer/'
        normalization_file_directory = normalization_file_directory + '/soccer/'

        training_file_names.append('Q_standard_shot_soccer')
        training_file_names.append('impact_standard_shot_soccer')

        training_file_names.append('Q_simple_pass_soccer')
        training_file_names.append('impact_simple_pass_soccer')

    max_depth = None

    minSplitInstances = 100

    training_epochs = None

    min_Q_difference_threshold = None
    number_of_gaps = None

    variance_reduction_prune = None

    # split_method = 0 # Q-gap (min threshold)
    min_Q_difference_threshold = 0.01

    # split_method = 5 # Q-gap (top)
    # split_method = 6 # Q-gap-gradient (top)
    number_of_gaps = 10

    # split_method = 1 # Gaussian Matrix
    # split_method = 2 # segmented/piecewise regression
    # split_method = 3 # calculate variance reduction incrementally
    # split_method = 4 # use t-test instead of variance reduction to select splits
    split_methods = [2,3,4,1]

    # model_type = 0 # null model (no tree)
    model_type = 1 # linear model
    # model_type = 2 # regular regression tree (average on leaves instead of linear model)

    # pruning_method = 0 # no pruning
    # pruning_lambs = [None]

    # pruning_method = 1 # pruning by variance reduction
    # variance_reduction_prune = 0.003

    # pruning_method = 2 # pruning by squared error and L1 sum(weights) 
    # pruning_lambs = [0, 0.01, 0.1, 0.3, 0.5]

    pruning_method = 3 # pruning by squared error and L0 len(weights)
    pruning_lambs = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01]

    # held_out_test_set = False
    held_out_test_set = True

    # make a set of all possible combinations with 1 element from each category
    # for example, [['a','b'],[1,2],[0.5, 0.6, 0.7]] becomes [('a', 1, 0.5), ('a', 1, 0.6), ('a', 1, 0.7), ('a', 2, 0.5), ('a', 2, 0.6), ('a', 2, 0.7), ('b', 1, 0.5), ('b', 1, 0.6), ('b', 1, 0.7), ('b', 2, 0.5), ('b', 2, 0.6), ('b', 2, 0.7)]
    combination_list = [training_file_names, split_methods, pruning_lambs]
    combinations = list(itertools.product(*combination_list))

    for combination in combinations:
        start_time = time.time()

        training_file_name = combination[0]
        split_method = combination[1]
        pruning_lamb = combination[2]

        print >> sys.stderr, 'start file {}.csv ... '.format(training_file_name)
        print >> sys.stderr, 'split_method: {}'.format(split_method)
        print >> sys.stderr, 'pruning_method: {}'.format(pruning_method)
        print >> sys.stderr, 'pruning_lamb: {}'.format(pruning_lamb)

        helper = Tree_Helper.Helper(save_directory=save_directory, problem=problem, training_file_name = training_file_name, minSplitInstances=minSplitInstances, Q_value_difference_significant_level = None, min_Q_difference_threshold = min_Q_difference_threshold, training_epochs=training_epochs, max_depth=max_depth, variance_reduction_prune=variance_reduction_prune, number_of_gaps=number_of_gaps, split_method=split_method, pruning_method=pruning_method, pruning_lamb=pruning_lamb, model_type=model_type, held_out_test_set=held_out_test_set, training_mode='')

        if 'impact_' in training_file_name:
            print_mode = 2 # print for Impact
        elif 'Q_' in training_file_name:
            print_mode = 1 # print for Q
        else:
            raise ValueError('Q or Impact?')

        file_name = training_file_path + '/' + training_file_name + '.csv'
        helper.episode(print_mode=print_mode, file_name=file_name, normalization_file_directory = normalization_file_directory, start_time=start_time)

    

    
