import sys

import glob
import os
import jax
from jax import config
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_platform_name", "gpu")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import joblib

import jaxley as jx
import jaxley.optimize.transforms as jt
from jaxley.connect import fully_connect, connect, sparse_connect, connectivity_matrix_connect
import optax
import pickle

import torch
from torch import optim
from sbi import utils as utils
from tqdm import tqdm

from network_utils import (make_network, set_train_parameters, get_currents, log_scale_forward, linear_scale_forward,
                           get_prior_dict, initialize_params)
from flow_utils import UniformPrior, PriorFiltered
from sklearn.linear_model import LinearRegression, Ridge

from neurodsp.spectral import compute_spectrum
import intrinsic_prior_configurations as prior_config

save_path = '/users/ntolley/data/ntolley/dendractor/intrinsic_permutations'

config_list = [
    ('Esoma_Isoma', prior_config.update_prior_dict_Esoma_Isoma), # 0
    ('Edend_Idend', prior_config.update_prior_dict_Edend_Idend), # 1
    ('Esoma_Idend', prior_config.update_prior_dict_Esoma_Idend), # 2
    ('Edend_Isoma', prior_config.update_prior_dict_Edend_Isoma), # 3
    # ('Esoma_Isomadend', prior_config.update_prior_dict_Esoma_Isomadend), # 4
    # ('Edend_Isomadend', prior_config.update_prior_dict_Edend_Isomadend), # 5
    # ('Esomadend_Isoma', prior_config.update_prior_dict_Esomadend_Isoma), # 6
    # ('Esomadend_Idend', prior_config.update_prior_dict_Esomadend_Idend), # 7
    ('Esomadend_Isomadend', prior_config.update_prior_dict_Esomadend_Isomadend) # 8
    ]

def simulate_sweep(theta, params, cue_currents, context_currents, seed):
    seed_key = jax.random.split(jax.random.PRNGKey(seed), num=4)
    rng = np.random.default_rng(seed=123)

    key_order = ["cue_ampa_gS", "context_ampa_gS",
                 "IE_gaba_gS", "II_gaba_gS", "EI_ampa_gS", "EE_ampa_gS",
                 "cue_dend_ampa_gS", "context_dend_ampa_gS",
                 "IE_dend_gaba_gS", "EE_dend_ampa_gS",
                 "cue_ampa_pconn", "context_ampa_pconn",
                 "IE_gaba_pconn", "II_gaba_pconn", "EI_ampa_pconn", "EE_ampa_pconn",
                 "cue_dend_ampa_pconn", "context_dend_ampa_pconn",
                 "IE_dend_gaba_pconn", "EE_dend_ampa_pconn",
                 "E_Leak_gLeak", "E_dend_Leak_gLeak", "I_Leak_gLeak",
                 "E_Km_gKm", "E_CaL_gCaL", "E_CaT_gCaT", "I_Km_gKm", "I_CaL_gCaL", "I_CaT_gCaT",
                 "E_dend_Km_gKm", "E_dend_CaL_gCaL", "E_dend_CaT_gCaT",]

    # params is a list of single element dicitonaries, this is to just find the index
    key_mapping = {list(param_dict.keys())[0]: idx for idx, param_dict in enumerate(params)}
    theta_dict = {param_name: prior_dict[param_name]['rescale_function'](
        theta[param_idx], prior_dict[param_name]['bounds']) for 
        param_idx, param_name in enumerate(key_order)}

    # Need to treat connections with special care
    # First create vector with identicial conductances for every synapse
    # Then mask out connections based on their probability
    for conn_name in ["cue_ampa", "context_ampa", "cue_dend_ampa", "context_dend_ampa",
                      "IE_gaba", "II_gaba", "EI_ampa", "EE_ampa", "IE_dend_gaba", "EE_dend_ampa"]:
        conn_g_name = f'{conn_name}_gS'
        conn_prob_name = f'{conn_name}_pconn'
        key_idx = key_mapping[conn_g_name]
        num_vals = len(params[key_idx][conn_g_name])

        new_vals = np.repeat(theta_dict[conn_g_name], num_vals)
        mask = rng.uniform(0, 1, size=num_vals) < theta_dict[conn_prob_name]
        new_vals = new_vals * mask

        params[key_idx][conn_g_name] = new_vals

    # No prob masking for biophysics, just update param vectors
    for param_name in ["E_Leak_gLeak", "E_dend_Leak_gLeak", "I_Leak_gLeak",
                       "E_Km_gKm", "E_CaL_gCaL", "E_CaT_gCaT",
                       "I_Km_gKm", "I_CaL_gCaL", "I_CaT_gCaT",
                       "E_dend_Km_gKm", "E_dend_CaL_gCaL", "E_dend_CaT_gCaT",]:
        key_idx = key_mapping[param_name]
        num_vals = len(params[key_idx][param_name])

        new_vals = np.repeat(theta_dict[param_name], num_vals)
        params[key_idx][param_name] = new_vals


    net.delete_stimuli()
    
    noise_scale = 0.06
    cue_noise = jax.random.normal(key=seed_key[0], shape=cue_currents.shape) * noise_scale
    context_noise = jax.random.normal(key=seed_key[1], shape=context_currents.shape) * noise_scale


    # Only add noise during stim period
    # cue_noise = jnp.zeros(shape=cue_currents.shape)
    # context_noise = jnp.zeros(shape=context_currents.shape)
    # stim_len = 1000
    # # cue_start = 0
    # cue_start = 28000
    # cue_stop = cue_start + stim_len
    # cue_noise = cue_noise.at[:, cue_start:cue_stop].set(
    #     jax.random.normal(key=seed_key[1], shape=(context_currents.shape[0], stim_len)) * noise_scale)
        
    # # context_start = 0
    # context_start = 10000
    # context_stop = context_start + stim_len
    # context_noise = context_noise.at[:, context_start:context_stop].set(
    #     jax.random.normal(key=seed_key[2], shape=(context_currents.shape[0], stim_len)) * noise_scale)

    # Attach stimulation
    data_stimuli = net.cell(list(gid_ranges['cue'])).branch(0).comp(0).data_stimulate(cue_currents + cue_noise)
    data_stimuli = net.cell(list(gid_ranges['context'])).branch(0).comp(0).data_stimulate(
        context_currents + context_noise, data_stimuli=data_stimuli)

    vmin, vmax = -80, -40
    E_voltages = jax.random.uniform(key=seed_key[2], minval=vmin, maxval=vmax, shape=(len(net.cell(list(gid_ranges['E'])).nodes),))
    I_voltages = jax.random.uniform(key=seed_key[3], minval=vmin, maxval=vmax, shape=(len(net.cell(list(gid_ranges['I'])).nodes),))

    param_state = None
    param_state = net.cell(list(gid_ranges['E'])).data_set('v', E_voltages, param_state)
    param_state = net.cell(list(gid_ranges['I'])).data_set('v', I_voltages, param_state)


    net.delete_recordings()
    net.branch(0).comp(0).record('v')

    s = jx.integrate(net, t_max=t_max, params=params, data_stimuli=data_stimuli, param_state=param_state, delta_t=dt)
    return s

def get_opt_data(data_path):
    print(f'Loading data from: {data_path}')
    theta_list = list()
    error_list = list()

    num_flows = 1
    for flow_idx in range(num_flows):
        print(f'Flow {flow_idx}')
        theta = np.load(f'{data_path}/theta_{flow_idx}.npy')
        error = np.load(f'{data_path}/flow_error_{flow_idx}.npy')


        rate_gids = list(gid_ranges['E_rate']) + list(gid_ranges['I_rate'])
        voltage_gids = list(gid_ranges['E'])

        theta_list.append(theta)
        error_list.append(error)


    error_sort = np.argsort(error)

    res_dict = {'theta_list': theta_list, 'error_list': error_list, 'error_sort': error_sort, 
                }

    return res_dict



if __name__ == "__main__":
    flow_idx = 0 # flow used for random init simulations

    dt = 0.025
    t_max = 1000
    time_vec = jnp.arange(0, t_max, dt)

    downsample_factor = 10
    dt_flow = dt * downsample_factor
    fs_flow = (1/dt_flow) * 1e3
    time_vec_flow = np.arange(0, t_max, dt_flow)
    burn_in = int(8000 / downsample_factor)

    for config_name, update_prior_dict in config_list:
        data_path = f'{save_path}/{config_name}'

        with open(f'{data_path}/jaxley_net.pkl', 'rb') as f:
            net, gid_ranges = pickle.load(f)

        res_dict = get_opt_data(data_path)
        theta = res_dict['theta_list'][flow_idx]
        theta_sort = np.argsort(res_dict['error_list'][flow_idx])
        theta_idx = theta_sort[0]
        print(f"Starting error: {res_dict['error_list'][flow_idx][theta_idx]}")

        num_E_cells, num_I_cells = len(gid_ranges['E']), len(gid_ranges['I'])
        num_cue_cells = len(gid_ranges['cue'])

        params, _ = set_train_parameters(net, gid_ranges)
        prior_dict = get_prior_dict()
        update_prior_dict(prior_dict)

        input_list = jnp.array([[-2,-2,1], [2,2,1], [-2, 2,1], [2,-2,1],
                                [-2,-2,-1], [2,2,-1], [-2, 2,-1], [2,-2,-1]])
        # input_list = jnp.array([[-2,-2,1], [2,2,1],
        #                         [-2,-2,-1], [2,2,-1]])
        num_cond = input_list.shape[0]
        input_data = [get_currents(input_list[idx], gid_ranges, t_max, dt) for idx in range(num_cond)]
        cue_currents = jnp.stack([input_data[idx][0] for idx in range(num_cond)])
        context_currents = jnp.stack([input_data[idx][1] for idx in range(num_cond)])
        targets = np.concatenate([input_data[idx][2][:2, ::downsample_factor] for idx in range(num_cond)], axis=1).T
        targets_stacked = jnp.stack([input_data[idx][2][:2, ::downsample_factor] for idx in range(num_cond)])

        batch_size = 5
        cue_currents_batch = jnp.tile(cue_currents, (batch_size, 1, 1))
        context_currents_batch = jnp.tile(context_currents, (batch_size, 1, 1))
        print(cue_currents_batch.shape)

        jitted_simulate = jit(simulate_sweep)
        jitted_vmapped_simulate = vmap(jitted_simulate, in_axes=(0, None, 0, 0, 0))

        best_theta = theta[theta_idx:theta_idx+1, :]
        
        # Run simulations in batch
        num_random_init = 10
        seed_array = jnp.arange(0, num_cond*num_random_init)

        output_list = list()
        for start_idx in range(0, num_random_init, batch_size):
            print(f'Batch: {start_idx}')
            end_idx = np.min([start_idx + batch_size, num_random_init])

            theta_batch = jnp.repeat(best_theta, num_cond * (end_idx - start_idx), axis=0)
            seed_batch = seed_array[start_idx*num_cond:end_idx*num_cond]

            output = np.array(jitted_vmapped_simulate(theta_batch, params, cue_currents_batch, context_currents_batch, seed_batch))
            output = output[:, :, ::downsample_factor]
            output_list.append(output)
        output_array = np.concatenate(output_list)

        random_init_dict = {
            'name': config_name,
            'output_array': output_array,
            'targets': targets,
            'targets_stacked': targets_stacked,
            'input_list': input_list,
            'theta': theta,
            'gid_ranges': gid_ranges
        }

        random_init_save_path = '/users/ntolley/data/ntolley/dendractor/intrinsic_permutations/random_initialization'
        fname = f'{config_name}_random_init.pkl'
        with open(f'{random_init_save_path}/{fname}', 'wb') as f:
            pickle.dump(random_init_dict,f)

