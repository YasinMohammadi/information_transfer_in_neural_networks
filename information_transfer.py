###############################################
#                    Yasin Mk
#            information transfer in networks
#                   fall 2022
#                Shiraz university



#import libraries
import networkx
import numpy
import scipy
import pandas
import pyinform
import math
#https://elife-asu.github.io/PyInform/timeseries.html#module-pyinform.transferentropy

import random

#functions

def refractory(new_fire_list_, fire_list_):
    refractory_list = numpy.multiply(new_fire_list_, fire_list_)
    new_fire_list_ = numpy.subtract(new_fire_list_, refractory_list)
    return new_fire_list_

def activate_network_no_refractory(fire_list_, adjacency_matrix_):
    fire_probability_list_ = fire_list_ * adjacency_matrix_
    network_size_ = len(fire_list_)
    chance_ = numpy.random.uniform(0 ,1 ,network_size_)   
    fire_list_ = (numpy.divide(fire_probability_list_, chance_) >= 1).astype(numpy.int8)
    return fire_list_

def activate_network_with_refractory(fire_list_, adjacency_matrix_):
    fire_probability_list_ = fire_list_ * adjacency_matrix_
    network_size_ = len(fire_list_)
    chance_ = numpy.random.uniform(0 ,1 ,network_size_)   
    new_fire_list_ = (numpy.divide(fire_probability_list_, chance_) >= 1).astype(numpy.int8)
    new_fire_list_ = refractory(new_fire_list_, fire_list_)
    return  new_fire_list_

def activate_network(fire_list_, adjacency_matrix_, refractory_):
    
    fire_list_ = activate_network_with_refractory(fire_list_, adjacency_matrix_) if refractory_ == 1 else activate_network_no_refractory(fire_list_, adjacency_matrix_)
    return fire_list_

def stabilize_network(fire_list_, adjacency_matrix_, stablize_threshold_, refractory_):
    network_size_ = len(fire_list_)
    history_ = numpy.zeros(network_size_)
    is_unstable = 1
    count_ = 0
    while(is_unstable):
        for iteration_ in range(network_size_):
            fire_list_ = activate_network(fire_list_, adjacency_matrix_, refractory_)
            history_[iteration_] = numpy.mean(fire_list_)
        if (numpy.var(history_) < stablize_threshold_):
            is_unstable = 0
        count_ += 1
        if (count_ > network_size_*100):
            break
    return fire_list_

def lambda_from_sigma(network_size_, inclusion_probability_, sigma_, inhibition_probability_):
    mean_K_ = network_size_ * inclusion_probability_
    lambda_= sigma_ * (mean_K_ * (1 - 2* inhibition_probability_))
    return lambda_

def initial_fireing_condotion(network_size_, firing_probationary_):
    initial_fireing_ = numpy.random.choice([0, 1], size=network_size_, p=[(1-firing_probationary_), firing_probationary_])
    return initial_fireing_

def inhibition_condotion(network_size_, inhibition_probability_):
    inhibition_list_ = numpy.random.choice([1, -1], size=network_size_, p=[(1-inhibition_probability_), inhibition_probability_])
    return inhibition_list_

def mean_activity_calc(fire_list_, adjacency_matrix_, refractory_):

    network_size_ = len(fire_list_)
    sum_ = 0
    for iteration_ in range(network_size_):
        sum_ += numpy.mean(activate_network(fire_list_, adjacency_matrix_, refractory_))
    mean_activity_ = sum_ / network_size_    

    return mean_activity_

def avalanche_size_probability_distribution(fire_list_, adjacency_matrix_, refractory_, mean_activity_, avalanch_runtime_):

    network_size_ = len(fire_list_)
    avalanche_count_list_ = numpy.zeros(network_size_)
    avalanche_list_ = []
    for iteration_ in range(avalanch_runtime_):
        avalanche_size = 0
        fire_list_ = activate_network(fire_list_, adjacency_matrix_, refractory_)
        activity_ = numpy.mean(fire_list_)

        if (activity_ > mean_activity_):
            avalanche_size = avalanche_size + activity_ - mean_activity
            while (activity_ > mean_activity_):
                avalanche_size = avalanche_size + activity_ - mean_activity
                fire_list_ = activate_network(fire_list_, adjacency_matrix_, refractory_)
                activity_ = numpy.mean(fire_list_)

            avalanche_list_.append(avalanche_size)
        # else:
        #      avalanche_list_.append(0)
    avalanche_list_ = numpy.array(avalanche_list_)
    bin_start_ = math.log10((numpy.min(avalanche_list_))) 
    bin_end_ = math.log10((numpy.max(avalanche_list_)))
    # bin_end_ = numpy.max(avalanche_list_)
    bins_ = numpy.logspace(bin_start_, bin_end_, num=30, endpoint=True, base=10.0, dtype=None, axis=0)
    avalanche_hist_list_ = numpy.histogram(avalanche_list_, bins=bins_, range=None, density=None, weights=None)

    # avalanche_freq_list_ = scipy.stats.relfreq(avalanche_list_, numbins=1000, defaultreallimits=None, weights=None)       

    # return avalanche_freq_list_
    return avalanche_hist_list_ 

def avalanche_area_probability_distribution(fire_list_, adjacency_matrix_, refractory_, mean_activity_, avalanch_runtime_):

    avalanche_list_ = []
    network_size_ = len(fire_list_)

    for iteration_ in range(avalanch_runtime_):
        avalanche_activity_ = numpy.zeros(network_size_)
        fire_list_ = activate_network(fire_list_, adjacency_matrix_, refractory_)
        activity_ = numpy.mean(fire_list_)

        if (activity_ > mean_activity_):
            avalanche_activity_ = numpy.add(avalanche_activity_, fire_list_)
            activity_ = numpy.mean(fire_list_)
            while (activity_ > mean_activity_):
                avalanche_activity_ = numpy.add(avalanche_activity_, fire_list_)
                fire_list_ = activate_network(fire_list_, adjacency_matrix_, refractory_)
                activity_ = numpy.mean(fire_list_)

            avalanche_size_ = numpy.count_nonzero(avalanche_activity_) / network_size_
            # avalanche_size_ = numpy.mean(avalanche_activity_)
            avalanche_list_.append(avalanche_size_)
            

    avalanche_list_ = numpy.array(avalanche_list_)
    bin_start_ = 0
    bin_end_ = numpy.max(avalanche_list_)
    bins_ = numpy.logspace(bin_start_, bin_end_, num=100, endpoint=True, base=10.0, dtype=None, axis=0)
    avalanche_hist_list_ = numpy.histogram(avalanche_list_, bins=bins_, range=None, density=None, weights=None)      

    # return avalanche_list_
    return avalanche_hist_list_ 

def transfer_entropy(fire_list_, adjacency_matrix_, refractory_, te_runtime_, te_history_length_):

    transfer_entropy_list_ = []
    for iteration_ in range(te_runtime_):
        new_fire_list_ = (activate_network(fire_list_, adjacency_matrix_, refractory_))
        transfer_entropy_ = pyinform.transferentropy.transfer_entropy(new_fire_list_, fire_list_, te_history_length_, condition=None, local=False)
        transfer_entropy_list_.append(transfer_entropy_)
        fire_list_ = new_fire_list_
    
    transfer_entropy_list_ = numpy.array(transfer_entropy_list_)
    return transfer_entropy_list_

def monte_carlo_transfer_entropy(history_, kullback_history_):


    try:
        history_length_ = history_.shape[1]
    except:
        print('history length is undefined. try agin.')
        pass
    if(history_length_ == 0):
        print('history length is zero. try agin.')
        pass

    if((history_length_ - kullback_history_) <= 0):
        print('Not enough history. try gathereting more information')
        pass


    network_size_ = history_.shape[0]
    # history_size_ = history_.shape[1]
    mean_transfer_entropy_ = 0
    
    monte_carlo_divergence = 1
    counter_ = 0
    while(monte_carlo_divergence):
        # history_row_ =random.randint(kullback_history_,network_size)
        source_num_ = random.randint(0,network_size)
        target_num_  = random.randint(0,network_size)

        source_ = history_[:, source_num_]  
        target_ = history_[:, target_num_]


        transfer_entropy_ = pyinform.transferentropy.transfer_entropy(source_, target_, kullback_history_, condition=None, local=False)
        mean_transfer_entropy_ = (mean_transfer_entropy_ * counter_ + transfer_entropy_) / (counter_ + 1)
        counter_ = counter_ + 1


        if(counter_>=200):
            monte_carlo_divergence = 0
        # try:
        #     transfer_entropy_ = pyinform.transferentropy.transfer_entropy(source_, target_, kullback_history_, condition=None, local=False)
        #     mean_transfer_entropy = (mean_transfer_entropy * counter_ + transfer_entropy_) / (counter_ + 1)
        #     counter_ = counter_ + 1
        # except:
        #     print('transfer entropy couldn\'t resolve')


    return mean_transfer_entropy_





# Network Properties

# network_size = 8192
connection_probability = 0.002
# inhibition_probability = 0.2
# refractory_period = 0
directed_status = True

# Avalanch Properties of Network
# avalanch_runtime = network_size * 100

# Network Properties for Lobster graph
backbone_probability = 0.15             # Probability of adding an edge to the backbone
beyond_backbone_probability = 0.20      # Probability of adding an edge one level beyond backbone

# Network Properties of geographical threshold graph
threshold = 0.001#1 / (connection_probability * network_size)

# Fire Properties
# critical_sigma = 1 / (network_size*connection_probability*(1-2*inhibition_probability))
# sigma_step = critical_sigma * 0.05
# steps = 2
# min_sigma = critical_sigma * 0.0
# max_sigma = critical_sigma * 3.0
# initial_firing_probationary = 0.1
stablize_threshold = 0.005
mean_degree = 20


# Transfer Entropy Properties
te_runtime = 1000
te_history_length = 2


#creating dataframe
columns_ = ['graph type','nodes', 'sigma','lambda', 'inhibition probability', 'connection probability', 'refractory period',
                                                'mean activity', 'transfer entropy']
parameter_dataframe = pandas.DataFrame(columns=columns_)


# Creat Network

# Efficiently returns a Erdős-Rényi graph genetation in order of O(m+n)
# simulation_graph = networkx.fast_gnp_random_graph(network_size, connection_probability, seed=None, directed=directed_status)
# Returns a random graph using Barabási–Albert preferential attachment
# barabashi_graph = networkx.barabasi_albert_graph(network_size, connection_probability, seed=None, initial_graph=None)
# Returns a random lobster graph
# lobster_graph = networkx.random_lobster(network_size, backbone_probability, beyond_backbone_probability, seed=None)
# Returns a geographical threshold graph
# geo_graph = networkx.geographical_threshold_graph(network_size, threshold, dim=2, pos=None, weight=None, metric=None, p_dist=None, seed=None)
# Returns a Watts–Strogatz small-world graph.
# w_s_graph = networkx.watts_strogatz_graph(network_size, mean_degree, p, seed=None)



refractory_period_list = [1]
# network_size_list = [ 2**10, int(2**10*1.5), 2**11]
dorogovtsev_generation = [7, 8, 9]
inhibition_percent_probability_list = [20]
index = 0
for refractory_period in refractory_period_list:
    for generation in dorogovtsev_generation:
        for inhibition_percent in inhibition_percent_probability_list:

            # Efficiently returns a Erdős-Rényi graph genetation in order of O(m+n)
            simulation_graph = networkx.dorogovtsev_goltsev_mendes_graph(generation, create_using=None)

            network_size = simulation_graph.number_of_nodes()
            
            inhibition_probability = inhibition_percent / 100
            avalanch_runtime = network_size * 10

            # Fire Properties
            critical_sigma = 1 / (network_size*connection_probability*(1-2*inhibition_probability))
            sigma_step = critical_sigma * 0.1
            steps = 2
            min_sigma = critical_sigma * 0.0
            max_sigma = critical_sigma * 5
            initial_firing_probationary = 0.05


            
            for sigma in numpy.arange(min_sigma ,max_sigma ,sigma_step):

                # Weight Networks
                inhibition_list = inhibition_condotion(network_size, inhibition_probability)
                initial_fireing = initial_fireing_condotion(network_size, initial_firing_probationary)
                for (u, v) in simulation_graph.edges():
                    simulation_graph.edges[u,v]['weight'] = numpy.random.uniform(0, 2*sigma) * inhibition_list[u]


                # network to sparse matrix
                simulation_graph_sparse = networkx.to_scipy_sparse_matrix(simulation_graph)
                # Largest Eig
                lambda_, vec = scipy.sparse.linalg.eigs(simulation_graph_sparse, k=1) 
                # initial activation
                er_fire_list = activate_network(initial_fireing, simulation_graph_sparse, refractory_period)

                
                # stabilizing the network
                for i in range(network_size):
                    er_fire_list = activate_network(er_fire_list, simulation_graph_sparse, refractory_period)
                    
                er_fire_list = stabilize_network(er_fire_list, simulation_graph_sparse, stablize_threshold, refractory_period)

                # Mean Activity Calc
                mean_activity = mean_activity_calc(er_fire_list, simulation_graph_sparse, refractory_period)

                # file_name = ('ER-avalanche-area'+str(network_size) + ('-') + str(lambda_) + ('-')+ str(mean_activity) + ('-')+str(inhibition_probability)+ (str(refractory_period))+'.csv')

                # TE Calc
                transfer_entropy_list = transfer_entropy(er_fire_list, simulation_graph_sparse, refractory_period, te_runtime, te_history_length)
                mean_transfer_entropy = numpy.mean(transfer_entropy_list)

                # Avalance calc
                # if (mean_activity != 0):
                #     avalanche_probability_distribution_list = avalanche_size_probability_distribution(er_fire_list, simulation_graph_sparse, refractory_period, mean_activity, avalanch_runtime)                
                #     pandas.DataFrame(avalanche_probability_distribution_list).to_csv(file_name, header=None, index=None)
                #     # avalanche_probability_distribution_list.frequency.tofile('data2.csv', sep = ',')
                #     print(avalanche_probability_distribution_list)

                list_ = ['dorogovtsev', network_size, sigma, float(lambda_.real), inhibition_probability, connection_probability, refractory_period, mean_activity, mean_transfer_entropy]
                parameter_dataframe.loc[index] = list_

                index += 1

                parameter_dataframe.to_csv('phase transition test.csv')

# parameter_dataframe.to_csv('phase transition test.csv')

