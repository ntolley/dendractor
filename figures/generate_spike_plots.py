import sys
sys.path.append('../code')
sys.path.append('../externals/SIMNETS-Python/')

import os

import jax
import jax.numpy as jnp
import jaxley as jx

import matplotlib.pyplot as plt
import numpy as np
from network_utils import (make_network, set_train_parameters, gaussian_tuning,
                           StimSynapse, get_currents, IonotropicSynapse, get_prior_dict)
from jax import config
import pickle
from networkx import connected_watts_strogatz_graph, adjacency_matrix,gaussian_random_partition_graph
# from jaxley_mech.synapses.destexhe98 import AMPA, GABAa, GABAb, NMDA

import pandas as pd
import seaborn as sns

from neurodsp.spectral import compute_spectrum
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering

from tqdm import tqdm

import simnets

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
# config.update("jax_platform_name", "gpu")

if __name__ == "__main__":

    data_path =  '/users/ntolley/data/ntolley/dendractor/memory_permutations/random_initialization'
    figure_path = '/users/ntolley/data/ntolley/dendractor/memory_permutations/figures'

    file_list = os.listdir(data_path)

    random_init_dict = dict()
    for file_idx in tqdm(range(len(file_list))):
        with open(f'{data_path}/{file_list[file_idx]}', 'rb') as f:
            res_dict = pickle.load(f)
        
        random_init_dict[res_dict['name']] = res_dict


    config_names = [
        # 'cuesomaampa_Esomaampa_Edendnmda',
        # 'cuesomaampa_Esomanmda_Edendampa',
        # 'cuesomanmda_Esomaampa_Edendnmda',
        # 'cuesomanmda_Esomanmda_Edendampa',
        'cuesomanmda_Esomaampa_Edendampa', 
        'cuesomaampa_Esomaampa_Edendampa', 
        # 'cuedendampa_Esomaampa_Edendnmda',
        # 'cuedendampa_Esomanmda_Edendampa',
        # 'cuedendnmda_Esomaampa_Edendnmda',
        # 'cuedendnmda_Esomanmda_Edendampa',
        'cuedendnmda_Esomaampa_Edendampa',
        'cuedendampa_Esomaampa_Edendampa',
        ]

    input_names = ['(-1,-1)', '(1,1)', '(-1,1)', '(1,-1)']

    for name in config_names:
        res_dict = random_init_dict[name]

        output_array = res_dict['output_array']
        targets = res_dict['targets']
        gid_ranges = res_dict['gid_ranges']
        input_list = res_dict['input_list']
        num_cond = input_list.shape[0]
        print(name)

        t_max = 1000
        dt = 0.25 # simulation output downsampled by factor of 10
        fs = (1/dt) * 1e3
        time_vec = np.arange(0, t_max, dt)
        downsample_factor = 10
        burn_in = int(0 / downsample_factor)

        #_____________________________________
        # Generate spike raster plots
        print('Generating raster plots')

        fontsize = 15
        ticksize = 10
        labelsize=13
        threshold = 0.0

        spike_color_dict = {'E': '#37abc8', 'I': '#d35f5f', 'context': '#1f77b4', 'cue': '#2ca02c', 'noise_E': 'k', 'noise_I': 'k' }
        plt.figure(figsize=(14,5))
        for plot_idx, sim_idx in enumerate(range(num_cond)):
            # sim_idx *= num_cond
            # sim_idx += 10
            
            plt.subplot(2,4, plot_idx+1)
            s = output_array[sim_idx, :]
            above_threshold = s > threshold
            spike_gids, spike_times = np.where(np.diff(above_threshold.astype(int), axis=1) == 1)

            plot_cell_counter = 0
            plot_cells = ['E', 'I', 'cue']
            for type_idx, cell_type in enumerate(plot_cells):
                cell_range = gid_ranges[cell_type]
                mask = np.isin(spike_gids, cell_range)
                num_cells = len(cell_range)
                y_offset = plot_cell_counter - list(cell_range)[0]

                plt.scatter(spike_times[mask] * dt, -spike_gids[mask] - y_offset - type_idx * 10, label=name, s=1, alpha=0.5, color=spike_color_dict[cell_type])

                plot_cell_counter += num_cells
            plt.ylabel('Cell gid', fontsize=fontsize)
            plt.yticks([])
            plt.xlim(10, time_vec[-1])
            plt.title(input_names[plot_idx], fontsize=labelsize)
            plt.tight_layout()

            plt.xticks(fontsize=ticksize)
            plt.xlabel('Time (ms)', fontsize=fontsize)
        
        fpath = f'{figure_path}/spike_figures/spike_raster'
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(f'{fpath}/{name}_raster.png')

        # ____________________________________________
        # Fit simnets model
        print('Generating simnets plots')

        projection_args = {'n_components': 2, 'perplexity': 30, 'learning_rate': 5}
        st_args = (30,)
        # st_args = (1,)

        model1 = simnets.SIMNETS(st_dist='victor-purpura', unit_dist='euclidean',
                                projection='TSNE', st_args=st_args)

        cell_type_list = ['E', 'I', 'cue']
        st_data = list()
        st_data_labels = list()
        for cell_type in cell_type_list:
            for gid in gid_ranges[cell_type]:
                spike_times_list = list()
                spike_count = 0
                for sim_idx in range(output_array.shape[0]):
                    s = output_array[sim_idx, gid, :]
                    above_threshold = s > threshold
                    spike_times = (np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1).astype(float)
                    spike_times_list.append(spike_times)
                st_data.append(spike_times_list)
                st_data_labels.append(cell_type)
                    
                    
        model1.fit(st_data)
        _ = model1.project(projection_args=projection_args)


        # Identify best cluster num
        range_n_clusters = range(2,20)
        silhouette_score_list = list()
        for n_clusters in range_n_clusters:
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(model1.simnets_points)
            silhouette_avg = silhouette_score(model1.simnets_points, cluster_labels)
            silhouette_score_list.append(silhouette_avg)

        best_n_clusters = range_n_clusters[np.argmax(silhouette_score_list)]
        print(f'Best cluster num: {best_n_clusters}')

        clusterer = KMeans(n_clusters=best_n_clusters, random_state=10)
        best_cluster_labels = clusterer.fit_predict(model1.simnets_points)

        #_____________________________________________
        # Make TSNE SIMNETS plot
        labelsize = 13
        ticksize = 10
        plt.figure(figsize=(10, 3))
        plt.subplot(1,3,1)

        gid_offset = 0
        for cell_type in cell_type_list:
            cell_gids = np.array(list(gid_ranges[cell_type]))
            plot_gids = (cell_gids - cell_gids[0]) + gid_offset
            plt.scatter(model1.simnets_points[plot_gids,0], model1.simnets_points[plot_gids, 1],
                        label=cell_type, color=spike_color_dict[cell_type], alpha=0.7, s=10)
            gid_offset += len(gid_ranges[cell_type])
        # plt.legend(fontsize=13)
        plt.xlabel('TSNE1', fontsize=labelsize)
        plt.ylabel('TSNE2', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.subplot(1,3,2)
        for cluster_idx in np.unique(best_cluster_labels):
            plot_gids = best_cluster_labels == cluster_idx
            plt.scatter(model1.simnets_points[plot_gids,0], model1.simnets_points[plot_gids, 1],
                        label=f'Cluster {cluster_idx}', s=10)
        # plt.legend(fontsize=13)
        plt.xlabel('TSNE1', fontsize=labelsize)
        plt.ylabel('TSNE2', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)


        plt.subplot(1,3,3)
        plt.plot(range_n_clusters, silhouette_score_list, color='k')
        plt.axvline(best_n_clusters, color='r', linewidth=3, linestyle='--', label='Best Score', zorder=-1)
        plt.legend(fontsize=10)
        plt.xlabel('Num Clusters', fontsize=labelsize)
        plt.ylabel('Silhouette Score', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.tight_layout()

        fpath = f'{figure_path}/spike_figures/simnets'
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(f'{fpath}/{name}_simnets.png')


        #_____________________________________________
        # Make single neuron plots
        print('Generating single neuron plots')
        
        for cell_idx in range(len(st_data)):
            ticksize = 13
            labelsize = 14

            num_repeats = output_array.shape[0] // num_cond
            cond_order = [0,1,2,3]
            trial_indices = list()
            for cond_idx in range(num_cond):
                for repeat_idx in range(num_repeats):
                    trial_indices.append((repeat_idx * num_cond) + cond_idx)

            plt.figure(figsize=(6,4))

            for plot_idx, trial_idx in enumerate(trial_indices):  

                st_trial = st_data[cell_idx][trial_idx] * dt
                y_pos = np.repeat(plot_idx, len(st_trial))
                plt.scatter(st_trial, y_pos, color='k', s=0.5)
            plt.xlim(0, 1100)


            cond_colors = [0,1,2,3]
            cond_labels = ['(-1,-1)', '(1,1)', '(-1,1)', '(1,-1)']
            for cond_idx in range(num_cond):
                y_offset = 0
                label = None
                if cond_idx > 3:
                    y_offset = 10
                    label = cond_labels[cond_idx]

                y1 = num_repeats * cond_idx - 0.1 + y_offset
                y2 = y1 + num_repeats
                plt.fill_between(x=[250, 300], y1=y1, y2=y2, zorder=-0.9, alpha=0.3, color=f'C{cond_colors[cond_idx]}', label=label)

            _ = plt.yticks([])
            _ = plt.xticks(fontsize=ticksize)

            plt.xlabel('Time (ms)', fontsize=labelsize)
            plt.title(f'Neuron {cell_idx} ({st_data_labels[cell_idx]})', fontsize=labelsize)
            # plt.legend()
            plt.gca().invert_yaxis()


            fpath = f'{figure_path}/spike_figures/single_neuron/{name}'
            os.makedirs(fpath, exist_ok=True)
            plt.savefig(f'{fpath}/{name}_cell{cell_idx}_raster.png')
        
        plt.close('all')



