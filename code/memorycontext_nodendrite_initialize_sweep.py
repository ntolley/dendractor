import sys

import glob
import os
import jax
from jax import config
config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")
config.update("jax_platform_name", "gpu")

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

from network_utils import make_network, set_train_parameters, get_currents, log_scale_forward, linear_scale_forward
from flow_utils import UniformPrior, PriorFiltered
from sklearn.linear_model import LinearRegression, Ridge

from neurodsp.spectral import compute_spectrum


data_path = '/users/ntolley/data/ntolley/dendractor/memorycontext_nodendrite'

def get_prior_dict():
    prior_dict = {
        "IE_gaba_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "II_gaba_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "EI_ampa_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "EE_ampa_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},

        "IE_dend_gaba_gS": {'bounds': (-20, -20), 'rescale_function': log_scale_forward},
        "EE_dend_ampa_gS": {'bounds': (-20, -20), 'rescale_function': log_scale_forward},
        
        "IE_gaba_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "II_gaba_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "EI_ampa_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "EE_ampa_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},

        "IE_dend_gaba_pconn": {'bounds': (0, 0.0), 'rescale_function': linear_scale_forward},
        "EE_dend_ampa_pconn": {'bounds': (0, 0.0), 'rescale_function': linear_scale_forward},

        "E_Leak_gLeak": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "E_dend_Leak_gLeak": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "I_Leak_gLeak": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},

        'E_Km_gKm': {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        'E_CaL_gCaL': {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        'E_CaT_gCaT': {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        'I_Km_gKm': {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        'I_CaL_gCaL': {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        'I_CaT_gCaT': {'bounds': (-9, -2), 'rescale_function': log_scale_forward},  

        "E_dend_Km_gKm": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "E_dend_CaL_gCaL": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "E_dend_CaT_gCaT": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},    
        }
    
    return prior_dict

def initialize_params(prior_dict, params):
    key_mapping = {list(param_dict.keys())[0]: idx for idx, param_dict in enumerate(params)}
    theta_dict = {param_name: param_dict['rescale_function'](thetai[param_idx].numpy(), param_dict['bounds']) for 
                    param_idx, (param_name, param_dict) in enumerate(prior_dict.items())}

def simulate_sweep(theta, params, cue_currents, context_currents):
    key_order = ["IE_gaba_gS", "II_gaba_gS", "EI_ampa_gS", "EE_ampa_gS",
                 "IE_dend_gaba_gS", "EE_dend_ampa_gS",
                 "IE_gaba_pconn", "II_gaba_pconn", "EI_ampa_pconn", "EE_ampa_pconn",
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
    for conn_name in ["IE_gaba", "II_gaba", "EI_ampa", "EE_ampa", "IE_dend_gaba", "EE_dend_ampa"]:
        conn_g_name = f'{conn_name}_gS'
        conn_prob_name = f'{conn_name}_pconn'
        key_idx = key_mapping[conn_g_name]
        num_vals = len(params[key_idx][conn_g_name])

        new_vals = np.repeat(theta_dict[conn_g_name], num_vals)
        mask = np.random.uniform(0, 1, size=num_vals) < theta_dict[conn_prob_name]
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
    
    data_stimuli = net.cell(list(gid_ranges['cue'])).branch(0).comp(0).data_stimulate(cue_currents)
    data_stimuli = net.cell(list(gid_ranges['context'])).branch(0).comp(0).data_stimulate(context_currents, data_stimuli=data_stimuli)

    net.delete_recordings()
    net.branch(0).comp(0).record('v')
    s = jx.integrate(net, t_max=t_max, params=params, checkpoint_lengths=checkpoints, data_stimuli=data_stimuli)
    return s

if __name__ == "__main__":
    dt = 0.025
    t_max = 2000
    time_vec = jnp.arange(0, t_max, dt)

    downsample_factor = 10
    dt_flow = dt * downsample_factor
    fs_flow = (1/dt_flow) * 1e3
    time_vec_flow = np.arange(0, t_max, dt_flow)
    burn_in = int(8000 / downsample_factor)

    prior_dict = get_prior_dict()
    # Used to reduce GPU memory (passed to simulate function)
    levels = 2
    time_points = t_max // dt + 2
    checkpoints = [int(np.ceil(time_points**(1/levels))) for _ in range(levels)]

    # net, gid_ranges = make_network()
    # with open(f'{data_path}/jaxley_net.pkl', 'wb') as f:
    #     pickle.dump((net, gid_ranges),f)

    with open(f'{data_path}/jaxley_net.pkl', 'rb') as f:
        net, gid_ranges = pickle.load(f)

    num_E_cells, num_I_cells = len(gid_ranges['E']), len(gid_ranges['I'])
    num_cue_cells = len(gid_ranges['cue'])

    # prepare samples for parameter sweep
    params, _ = set_train_parameters(net, gid_ranges)

    num_simulations = 250
    # num_simulations = 50
    num_prior_fits = 5
    num_iter = 5000

    input_list = jnp.array([[-2,-2,1], [2,2,1], [-2, 2,1], [2,-2,1],
                            [-2,-2,-1], [2,2,-1], [-2, 2,-1], [2,-2,-1]])
    # input_list = jnp.array([[-2,-2,1]])
    num_cond = input_list.shape[0]
    input_data = [get_currents(input_list[idx], gid_ranges, t_max, dt) for idx in range(num_cond)]
    cue_currents = jnp.stack([input_data[idx][0] for idx in range(num_cond)])
    context_currents = jnp.stack([input_data[idx][1] for idx in range(num_cond)])
    # targets = np.concatenate([input_data[idx][2][:, ::downsample_factor] for idx in range(num_cond)], axis=1).T
    targets = np.concatenate([input_data[idx][2][:2, ::downsample_factor] for idx in range(num_cond)], axis=1).T

    batch_size = 2
    cue_currents_batch = jnp.tile(cue_currents, (batch_size, 1, 1))
    context_currents_batch = jnp.tile(context_currents, (batch_size, 1, 1))
    print(cue_currents_batch.shape)

    jitted_simulate = jit(simulate_sweep)
    jitted_vmapped_simulate = vmap(jitted_simulate, in_axes=(0, None, 0, 0))
    for flow_idx in range(num_prior_fits):
        if flow_idx == 0:
            prior = UniformPrior(parameters=list(prior_dict.keys()))
        else:
            prior = prior_filtered

        theta = jnp.array(prior.sample((num_simulations,)).numpy())

        np.save(f'{data_path}/theta_{flow_idx}.npy', theta)

        # Run simulations in batch
        fname_list = list()
        for start_idx in range(0, num_simulations, batch_size):
            print(f'Batch: {start_idx}')
            end_idx = np.min([start_idx + batch_size, num_simulations])
            theta_batch = theta[start_idx:end_idx, :]
            theta_batch = jnp.repeat(theta_batch, num_cond, axis=0)

            output = np.array(jitted_vmapped_simulate(theta_batch, params, cue_currents_batch, context_currents_batch))
            output = output[:, :, ::downsample_factor]

            fname = f'{data_path}/tmp/x_{start_idx}-{end_idx}.npy'
            fname_list.append(fname)
            np.save(fname, output)

        # Aggregate files and save
        output_list = list()
        for fname in fname_list:
            output_list.append(np.load(fname))
        output_array = np.concatenate(output_list)
        if flow_idx == num_prior_fits - 1:
            np.save(f'{data_path}/x_out_{flow_idx}.npy', output_array)

        # Clean up temp files
        files = glob.glob(f'{data_path}/tmp/*')
        for f in files:
            os.remove(f)

        # Train flow for new prior
        # rate_gids = list(gid_ranges['E_rate']) + list(gid_ranges['I_rate'])
        rate_gids = list(gid_ranges['E_rate'])

        rates = output_array[:, rate_gids, :]

        num_sims, num_neurons, num_samples = rates.shape

        rates_stacked = rates.reshape((num_sims * num_neurons, num_samples))
        freqs, spectrum = compute_spectrum(rates_stacked, fs=fs_flow, nperseg=fs_flow*5)
        spectrum = spectrum.reshape((num_sims, num_neurons, -1))

        freq_mask = np.logical_and(freqs > 10, freqs < 40)
        # total_mask = freqs < 1000
        avg_spectrum = np.mean(spectrum, axis=1)
        band_power = np.sum(avg_spectrum[:, freq_mask], axis=1)
        # total_power = np.sum(avg_spectrum[:, total_mask], axis=1)
        # band_power = band_power / total_power # normalize by total spectral power

        x_train = list()
        band_power_avg = list()
        for sim_idx in range(0, output_array.shape[0], num_cond):
            temp_list = list()
            for cond_idx in range(num_cond):
                temp_list.append(output_array[sim_idx + cond_idx, gid_ranges['E_rate'], :])
            x_train.append(np.concatenate(temp_list, axis=1).T)
            band_power_avg.append(np.mean(band_power[sim_idx:sim_idx+num_cond]))
        band_power_avg = np.array(band_power_avg)

        error_list = list()
        model = Ridge(alpha=2.0)
        for sim_idx in range(len(x_train)):
            rate_pred = x_train[sim_idx]
            rate_pred += np.random.uniform(-0.1, 0.1, size=rate_pred.shape)
            y_pred = model.fit(rate_pred[burn_in:, :], targets[burn_in:, :]).predict(rate_pred[burn_in:, :])
   

            error = np.mean(np.square(targets[burn_in:, :] - y_pred))
            error_list.append(error)
            if sim_idx % 100 == 0:
                print(sim_idx, end=' ')
        error_list = np.array(error_list)
        np.save(f'{data_path}/flow_error_{flow_idx}.npy', error_list)
        np.save(f'{data_path}/flow_band_power_{flow_idx}.npy', band_power_avg)

        band_power_threshold = np.quantile(band_power_avg, 0.7)
        error_threshold = np.quantile(error_list[band_power_avg > band_power_threshold], 0.3)
        mask = np.logical_and(band_power_avg > band_power_threshold, error_list < error_threshold)

        print(f'Error Threshold: {error_threshold}; Band Power Threshold: {band_power_threshold}')
        print(f'{np.sum(mask)} sims remaining')

        # Filter theta using feature masks, take top simulations that separate inputs
        theta_filter = np.array(theta[mask])
        prior_filtered = PriorFiltered(parameters=list(prior_dict.keys()))
        optimizer = optim.Adam(prior_filtered.flow.parameters())


        # Train flow
        num_iter = 5000
        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            loss = -prior_filtered.flow.log_prob(inputs=theta_filter).mean()
            loss.backward()
            optimizer.step()
        state_dict = prior_filtered.flow.state_dict()
        joblib.dump(state_dict, f'{data_path}/prior_filtered_flow_{flow_idx}.pkl')


