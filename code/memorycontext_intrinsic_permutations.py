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

def get_save_path():
    save_path = '/users/ntolley/data/ntolley/dendractor/intrinsic_permutations'
    return save_path

def get_config_list():
    config_list = [
        ('Esoma_Isoma', prior_config.update_prior_dict_Esoma_Isoma), # 0
        ('Edend_Idend', prior_config.update_prior_dict_Edend_Idend), # 1
        ('Esoma_Idend', prior_config.update_prior_dict_Esoma_Idend), # 2
        ('Edend_Isoma', prior_config.update_prior_dict_Edend_Isoma), # 3
        ('Esoma_Isomadend', prior_config.update_prior_dict_Esoma_Isomadend), # 4
        ('Edend_Isomadend', prior_config.update_prior_dict_Edend_Isomadend), # 5
        ('Esomadend_Isoma', prior_config.update_prior_dict_Esomadend_Isoma), # 6
        ('Esomadend_Idend', prior_config.update_prior_dict_Esomadend_Idend), # 7
        ('Esomadend_Isomadend', prior_config.update_prior_dict_Esomadend_Isomadend) # 8
        ]
    return config_list


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
    # cue_noise = jax.random.normal(key=seed_key[0], shape=cue_currents.shape) * noise_scale
    # context_noise = jax.random.normal(key=seed_key[1], shape=context_currents.shape) * noise_scale


    # Only add noise during stim period
    cue_noise = jnp.zeros(shape=cue_currents.shape)
    context_noise = jnp.zeros(shape=context_currents.shape)
    stim_len = 1000
    # cue_start = 0
    cue_start = 28000
    cue_stop = cue_start + stim_len
    cue_noise = cue_noise.at[:, cue_start:cue_stop].set(
        jax.random.normal(key=seed_key[1], shape=(context_currents.shape[0], stim_len)) * noise_scale)
        
    # context_start = 0
    context_start = 10000
    context_stop = context_start + stim_len
    context_noise = context_noise.at[:, context_start:context_stop].set(
        jax.random.normal(key=seed_key[2], shape=(context_currents.shape[0], stim_len)) * noise_scale)

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
    net.cell(list(gid_ranges['E_rate'])).branch(0).comp(0).record('v')

    s = jx.integrate(net, t_max=t_max, params=params, data_stimuli=data_stimuli, param_state=param_state, delta_t=dt)
    return s

if __name__ == "__main__":
    save_path = get_save_path()
    config_list = get_config_list()

    # Pull config name and prior dict update function
    job_id = int(sys.argv[1])
    config_name, update_prior_dict = config_list[job_id]
    print(f'Running {config_name}')

    # Set up folder paths
    data_path = f'{save_path}/{config_name}'
    os.makedirs(f'{data_path}/tmp', exist_ok=True)

    # dt = 0.05
    dt = 0.025
    # t_max = 2000
    t_max = 1000
    time_vec = jnp.arange(0, t_max, dt)

    # Number of samples before calculating error
    burn_in = 30_000

    downsample_factor = 10

    prior_dict = get_prior_dict()
    update_prior_dict(prior_dict)



    net, gid_ranges = make_network()
    with open(f'{data_path}/jaxley_net.pkl', 'wb') as f:
        pickle.dump((net, gid_ranges),f)

    num_E_cells, num_I_cells = len(gid_ranges['E']), len(gid_ranges['I'])
    num_cue_cells = len(gid_ranges['cue'])

    # prepare samples for parameter sweep
    params, _ = set_train_parameters(net, gid_ranges)

    # num_simulations = 250
    num_simulations = 100
    num_prior_fits = 5
    num_iter = 5000

    # batch_size = 5
    # num_repeats = 5

    batch_size = 5
    num_repeats = 10
    # input_list = jnp.array([[-2,-2,1], [2,2,1], [-2, 2,1], [2,-2,1],
    #                         [-2,-2,-1], [2,2,-1], [-2, 2,-1], [2,-2,-1]])
    input_list = jnp.array([[-2,-2,1], [2,2,1], [-2,-2,-1], [2,2,-1]])
    num_inputs = input_list.shape[0]
    input_list = jnp.tile(input_list, (num_repeats, 1))

    num_cond = input_list.shape[0]

    input_data = [get_currents(input_list[idx], gid_ranges, t_max, dt) for idx in range(num_cond)]
    cue_currents = jnp.stack([input_data[idx][0] for idx in range(num_cond)])
    context_currents = jnp.stack([input_data[idx][1] for idx in range(num_cond)])

    targets_list = np.array([input_data[idx][2][:2, burn_in::downsample_factor] for idx in range(num_cond)])

    cue_currents_batch = jnp.tile(cue_currents, (batch_size, 1, 1))
    context_currents_batch = jnp.tile(context_currents, (batch_size, 1, 1))
    print(cue_currents_batch.shape)

    jitted_simulate = jit(simulate_sweep)
    jitted_vmapped_simulate = vmap(jitted_simulate, in_axes=(0, None, 0, 0, 0))

    for flow_idx in range(num_prior_fits):
        if flow_idx == 0:
            prior = UniformPrior(parameters=list(prior_dict.keys()))
        else:
            prior = prior_filtered

        theta = jnp.array(prior.sample((num_simulations,)).numpy())

        np.save(f'{data_path}/theta_{flow_idx}.npy', theta)

        # Run simulations in batch
        error_list = list()
        model = Ridge(alpha=2.0)
        for start_idx in range(0, num_simulations, batch_size):
            end_idx = np.min([start_idx + batch_size, num_simulations])
            theta_batch = theta[start_idx:end_idx, :]
            theta_batch = jnp.repeat(theta_batch, num_cond, axis=0)

            seed_batch = jnp.arange(start_idx*num_cond, end_idx*num_cond)
            output = np.array(jitted_vmapped_simulate(theta_batch, params, cue_currents_batch, context_currents_batch, seed_batch))
            output = output[:, :, burn_in::downsample_factor]

            for batch_idx in range(0, batch_size):
                batch_offset = batch_idx * num_cond
                x_list = list()
                for cond_idx in range(num_cond):
                    x_list.append(output[cond_idx + batch_offset, :, :])
                x_list = np.array(x_list)
                # x_train = np.concatenate(x_train, axis=1).T

                temp_error_list = list()
                for val_start in range(0, num_cond, num_inputs):
                    val_stop = val_start + num_inputs
                    val_mask = np.zeros(num_cond).astype(bool)
                    val_mask[val_start:val_stop] = True
                    train_mask = ~val_mask

                    x_train = np.concatenate(x_list[train_mask], axis=1).T
                    x_val = np.concatenate(x_list[val_mask], axis=1).T
                    targets_train = np.concatenate(targets_list[train_mask], axis=1).T
                    targets_val = np.concatenate(targets_list[val_mask], axis=1).T

                    y_pred = model.fit(x_train, targets_train).predict(x_val)
                    y_pred[y_pred > 2.0] = 2.0 # hard threshold on max value
                    y_pred[y_pred < -2.0] = -2.0 # hard threshold on max value

                    temp_error = np.mean(np.square(targets_val - y_pred))
                    temp_error_list.append(temp_error)

                error = np.mean(temp_error_list)

                error_list.append(error)
                print(f'Batch {start_idx + batch_idx}; avg error: {error}; error_list: {temp_error_list}')

            # x_train1 = list()
            # for cond_idx in range(num_train):
            #     x_train1.append(output[cond_idx, :, :])
            # x_train1 = np.concatenate(x_train1, axis=1).T

            # x_train2 = list()
            # for cond_idx in range(num_train, num_cond):
            #     x_train2.append(output[cond_idx, :, :])
            # x_train2 = np.concatenate(x_train2, axis=1).T

            # y_pred2 = model.fit(x_train1[burn_in:, :], targets_train1[burn_in:, :]).predict(x_train2[burn_in:, :])
            # error2 = np.mean(np.square(targets_train2[burn_in:, :] - y_pred2))

            # y_pred1 = model.fit(x_train2[burn_in:, :], targets_train2[burn_in:, :]).predict(x_train1[burn_in:, :])
            # error1 = np.mean(np.square(targets_train1[burn_in:, :] - y_pred1))

            # error_list.append(np.mean([error1, error2]))

        # Train flow for new prior
        error_list = np.array(error_list)
        np.save(f'{data_path}/flow_error_{flow_idx}.npy', error_list)

        error_threshold = np.quantile(error_list, 0.1)
        mask = error_list < error_threshold

        print(f'Error Threshold: {error_threshold}')
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


