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
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder
from tqdm import tqdm

from network_utils import (make_network, set_train_parameters, get_currents_nocontext, log_scale_forward, linear_scale_forward,
                           get_prior_dict, initialize_params, get_parameter_names)
from flow_utils import UniformPrior, PriorFiltered
from sklearn.linear_model import LinearRegression, Ridge

from neurodsp.spectral import compute_spectrum
import extrinsic_prior_configurations as prior_config

def get_save_path():
    # save_path = '/users/ntolley/data/ntolley/dendractor/extrinsic_permutations_nocontext'
    save_path = '/users/ntolley/data/ntolley/dendractor/extrinsic_permutations_nocontext_lowcueprob_somanmda'
    return save_path

def get_config_list():
    config_list = [
        ('contextsoma_cuesoma', prior_config.update_prior_dict_contextsoma_cuesoma), # 0
        ('contextdend_cuedend', prior_config.update_prior_dict_contextdend_cuedend), # 1
        ('contextsoma_cuedend', prior_config.update_prior_dict_contextsoma_cuedend), # 2
        ('contextdend_cuesoma', prior_config.update_prior_dict_contextdend_cuesoma), # 3
        ]
    return config_list

def simulate_sweep(theta, params, cue_currents, context_currents, seed):
    seed_key = jax.random.split(jax.random.PRNGKey(seed), num=4)
    rng = np.random.default_rng(seed=123)

    key_order, conn_names, biophysics_names = get_parameter_names()

    # params is a list of single element dicitonaries, this is to just find the index
    key_mapping = {list(param_dict.keys())[0]: idx for idx, param_dict in enumerate(params)}
    theta_dict = {param_name: prior_dict[param_name]['rescale_function'](
        theta[param_idx], prior_dict[param_name]['bounds']) for 
        param_idx, param_name in enumerate(key_order)}

    # Need to treat connections with special care
    # First create vector with identicial conductances for every synapse
    # Then mask out connections based on their probability
    for conn_name in conn_names:
        conn_g_name = f'{conn_name}_gS'
        conn_prob_name = f'{conn_name}_pconn'
        key_idx = key_mapping[conn_g_name]
        num_vals = len(params[key_idx][conn_g_name])

        new_vals = np.repeat(theta_dict[conn_g_name], num_vals)
        mask = rng.uniform(0, 1, size=num_vals) < theta_dict[conn_prob_name]
        new_vals = new_vals * mask

        params[key_idx][conn_g_name] = new_vals

    # No prob masking for biophysics, just update param vectors
    for param_name in biophysics_names:
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
    cue_start = 10000
    cue_stop = cue_start + stim_len
    cue_noise = cue_noise.at[:, cue_start:cue_stop].set(
        jax.random.normal(key=seed_key[1], shape=(context_currents.shape[0], stim_len)) * noise_scale)

    cue_noise = cue_noise.at[:, 0:stim_len].set(
        jax.random.normal(key=seed_key[1], shape=(context_currents.shape[0], stim_len)) * noise_scale)
        
    context_start = 10000
    context_stop = context_start + stim_len
    context_noise = context_noise.at[:, context_start:context_stop].set(
        jax.random.normal(key=seed_key[2], shape=(context_currents.shape[0], stim_len)) * noise_scale)

    context_noise = context_noise.at[:, 0:stim_len].set(
        jax.random.normal(key=seed_key[2], shape=(context_currents.shape[0], stim_len)) * noise_scale)

    # Attach stimulation
    data_stimuli = net.cell(list(gid_ranges['cue'])).branch(0).comp(0).data_stimulate(cue_currents + cue_noise)
    # data_stimuli = net.cell(list(gid_ranges['context'])).branch(0).comp(0).data_stimulate(
    #     context_currents + context_noise, data_stimuli=data_stimuli)

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

    dt = 0.025
    t_max = 1000
    time_vec = jnp.arange(0, t_max, dt)

    # Number of samples before calculating error
    burn_in = 10_000

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

    num_simulations = 100

    num_prior_fits = 10
    num_iter = 5000

    batch_size = 10
    num_repeats = 5


    input_list = jnp.array([[-2,-2,1], [2,2,1], [-2, 2,1], [2,-2,1]])
    num_inputs = input_list.shape[0]
    input_list = jnp.tile(input_list, (num_repeats, 1))

    num_cond = input_list.shape[0]

    input_data = [get_currents_nocontext(input_list[idx], gid_ranges, t_max, dt) for idx in range(num_cond)]
    cue_currents = jnp.stack([input_data[idx][0] for idx in range(num_cond)])
    context_currents = jnp.stack([input_data[idx][1] for idx in range(num_cond)])

    targets_list = np.array([input_data[idx][2][:2, burn_in::downsample_factor] for idx in range(num_cond)])

    cue_currents_batch = jnp.tile(cue_currents, (batch_size, 1, 1))
    context_currents_batch = jnp.tile(context_currents, (batch_size, 1, 1))
    print(cue_currents_batch.shape)

    jitted_simulate = jit(simulate_sweep)
    jitted_vmapped_simulate = vmap(jitted_simulate, in_axes=(0, None, 0, 0, 0))

    # Set up SBI objects
    prior = UniformPrior(parameters=list(prior_dict.keys()))
    proposal = prior
    inference = NPE(prior)
    global_error_threshold = 1.0

    # Accumulate theta values to train on every round
    all_theta_list = list()
    all_error_list = list()

    for flow_idx in range(num_prior_fits):
        theta = jnp.array(proposal.sample((num_simulations,)).numpy())

        np.save(f'{data_path}/theta_{flow_idx}.npy', theta)

        # Run simulations in batch
        error_list = list()
        y_pred_list = list()
        model = Ridge(alpha=2.0)
        for start_idx in range(0, num_simulations, batch_size):
            end_idx = np.min([start_idx + batch_size, num_simulations])
            theta_batch = theta[start_idx:end_idx, :]
            theta_batch = jnp.repeat(theta_batch, num_cond, axis=0)

            seed_batch = jnp.arange(start_idx*num_cond, end_idx*num_cond)
            output = np.array(jitted_vmapped_simulate(theta_batch, params, cue_currents_batch, context_currents_batch, seed_batch))
            output = output[:, :, burn_in::downsample_factor]

            # Loop over each unique parameter set (theta)
            for batch_idx in range(0, batch_size):
                batch_offset = batch_idx * num_cond
                x_list = list()
                for cond_idx in range(num_cond):
                    x_list.append(output[cond_idx + batch_offset, :, :])
                x_list = np.array(x_list)
                # x_train = np.concatenate(x_train, axis=1).T

                # Cross validation loop
                temp_error_list = list()
                temp_y_pred_list = list()
                for val_start in range(0, num_cond, num_inputs):
                    val_stop = val_start + num_inputs
                    val_mask = np.zeros(num_cond).astype(bool)
                    val_mask[val_start:val_stop] = True
                    train_mask = ~val_mask

                    x_train = np.concatenate(x_list[train_mask], axis=1).T
                    x_val = np.concatenate(x_list[val_mask], axis=1).T
                    targets_train = np.concatenate(targets_list[train_mask], axis=1).T
                    targets_val = np.concatenate(targets_list[val_mask], axis=1).T

                    # Calculate errors
                    y_pred = model.fit(x_train, targets_train).predict(x_val)
                    temp_error = np.mean(np.square(targets_val - y_pred))
                    temp_error_list.append(temp_error)

                    # Calculate mean output of network
                    y_pred_cond = [np.mean(model.predict(x_val_cond.T), axis=0) for x_val_cond in x_list[val_mask]]
                    temp_y_pred_list.append(np.concatenate(y_pred_cond))

                # Update with average predicted output over (vector of size (num_inputs))
                y_pred_avg = np.mean(np.array(temp_y_pred_list), axis=0)
                y_pred_list.append(y_pred_avg)

                error = np.mean(temp_error_list)
                error_list.append(error)
                print(f'Batch {start_idx + batch_idx}; avg error: {error}')

        # Save batch simulation outputs
        error_list = np.array(error_list)
        np.save(f'{data_path}/flow_error_{flow_idx}.npy', error_list)

        # Heavily penalize chance level predictions
        y_pred_mask = np.mean(np.abs(y_pred_list), axis=1) < 0.2
        error_list[y_pred_mask] += 1e3
        print(f'{np.sum(y_pred_mask)} null simulations')

        error_threshold = np.quantile(error_list, 0.1)
        if error_threshold < global_error_threshold:
            error_threshold = global_error_threshold

        all_error_list.append(error_list)
        all_theta_list.append(theta)

        error_train = np.concatenate(all_error_list)
        error_mask = error_train < error_threshold

        theta_train = np.concatenate(all_theta_list, axis=0)
        theta_train = theta_train[error_mask, :]
        print(f'Theta Train Shape: {theta_train.shape}')
        

        print(f'Error Threshold: {error_threshold}')


        # Filter theta using feature masks, take top simulations that separate inputs
        proposal = PriorFiltered(parameters=list(prior_dict.keys()))
        optimizer = optim.Adam(proposal.flow.parameters())

        # Train flow
        num_iter = 5000
        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            loss = -proposal.flow.log_prob(inputs=theta_train).mean()
            loss.backward()
            optimizer.step()
        state_dict = proposal.flow.state_dict()
        joblib.dump(state_dict, f'{data_path}/prior_filtered_flow_{flow_idx}.pkl')


