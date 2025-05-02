import sys
sys.path.append('../code')

import jax
from jax import jit, vmap
import jax.numpy as jnp
import jaxley as jx

import matplotlib.pyplot as plt
import numpy as np
from network_utils import make_network, set_train_parameters, gaussian_tuning, StimSynapse, get_currents, IonotropicSynapse
from jax import config
import pickle
from networkx import connected_watts_strogatz_graph, adjacency_matrix,gaussian_random_partition_graph

import pandas as pd
import seaborn as sns

from neurodsp.spectral import compute_spectrum
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA

config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")
config.update("jax_platform_name", "gpu")

from memorycontext_cuesoma_contextsoma import get_prior_dict

from memorycontext_cuesoma_contextsoma import update_prior_dict_cuesoma_contextsoma
from memorycontext_cuedend_contextdend import update_prior_dict_cuedend_contextdend
from memorycontext_cuesoma_contextdend import update_prior_dict_cuesoma_contextdend
from memorycontext_cuedend_contextsoma import update_prior_dict_cuedend_contextsoma

from memorycontext_cuesoma_contextsoma_cellsoma import update_prior_dict_cuesoma_contextsoma_cellsoma
from memorycontext_cuedend_contextdend_celldend import update_prior_dict_cuedend_contextdend_celldend
from memorycontext_cuesoma_contextsoma_celldend import update_prior_dict_cuesoma_contextsoma_celldend



def simulate_sweep(theta, params, cue_currents, context_currents, random_init=False, seed=123):
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

    # Voltage range for random initialization
    vmin, vmax = -80, -40
    E_voltages = np.random.uniform(vmin, vmax, size=len(net.cell(list(gid_ranges['E'])).nodes))
    I_voltages = np.random.uniform(vmin, vmax, size=len(net.cell(list(gid_ranges['I'])).nodes))
    net.cell(list(gid_ranges['E'])).set('v', E_voltages)
    net.cell(list(gid_ranges['I'])).set('v', I_voltages)
    s = jx.integrate(net, t_max=t_max, params=params, checkpoint_lengths=checkpoints, data_stimuli=data_stimuli)
    return s

def get_opt_data(data_path):
    print(f'Loading data from: {data_path}')
    theta_list = list()
    error_list = list()
    band_power_list = list()

    num_flows = 5
    for flow_idx in range(num_flows):
        print(f'Flow {flow_idx}')
        theta = np.load(f'{data_path}/theta_{flow_idx}.npy')
        error = np.load(f'{data_path}/flow_error_{flow_idx}.npy')
        band_power_avg = np.load(f'{data_path}/flow_band_power_{flow_idx}.npy')


        rate_gids = list(gid_ranges['E_rate']) + list(gid_ranges['I_rate'])
        voltage_gids = list(gid_ranges['E'])

        theta_list.append(theta)
        error_list.append(error)
        band_power_list.append(band_power_avg)


    error_sort = np.argsort(error)

    res_dict = {'theta_list': theta_list, 'error_list': error_list, 'band_power_list': band_power_list, 'error_sort': error_sort, 
                }

    return res_dict



if __name__ == "__main__":
    num_E_cells = 100
    num_I_cells = 50
    num_context_cells = 50
    num_cue_cells = 50

    net_dict = {
        'E': {'num_cells': num_E_cells},
        'I': {'num_cells': num_I_cells},
        'context': {'num_cells': num_context_cells},
        'cue': {'num_cells': num_cue_cells},
        'E_rate': {'num_cells': num_E_cells},
        'I_rate': {'num_cells': num_I_cells},
    }

    gid_ranges = dict()
    cell_count = 0
    for name, cell_dict in net_dict.items():
        num_cells = cell_dict['num_cells']
        gid_ranges[name] = range(cell_count, cell_count + num_cells)
        cell_count += num_cells

    data_path_cuesoma_contextsoma = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuesoma_contextsoma/'
    data_path_cuedend_contextdend = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuedend_contextdend/'
    data_path_cuesoma_contextdend = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuesoma_contextdend/'
    data_path_cuedend_contextsoma = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuedend_contextsoma/'
    data_path_cuesoma_contextsoma_cellsoma = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuesoma_contextsoma_cellsoma/'
    data_path_cuedend_contextdend_celldend = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuedend_contextdend_celldend/'
    data_path_cuesoma_contextsoma_celldend = '/users/ntolley/data/ntolley/dendractor/memorycontext_cuesoma_contextsoma_celldend/'

    dt = 0.025
    t_max = 2000
    time_vec = jnp.arange(0, t_max, dt)

    downsample_factor = 10
    dt_flow = dt * downsample_factor
    fs_flow = (1/dt_flow) * 1e3
    time_vec_flow = np.arange(0, t_max, dt_flow)
    burn_in = int(8000 / downsample_factor)

    # Used to reduce GPU memory (passed to simulate function)
    levels = 2
    time_points = t_max // dt + 2
    checkpoints = [int(np.ceil(time_points**(1/levels))) for _ in range(levels)]

    res_paths = {
        # 'cuesoma_contextsoma': {'data_path': data_path_cuesoma_contextsoma, 'update_prior': update_prior_dict_cuesoma_contextsoma, 'flow_idx': 4},
        # 'cuedend_contextdend': {'data_path': data_path_cuedend_contextdend, 'update_prior': update_prior_dict_cuedend_contextdend, 'flow_idx': 4},
        # 'cuesoma_contextdend': {'data_path': data_path_cuesoma_contextdend, 'update_prior': update_prior_dict_cuesoma_contextdend, 'flow_idx': 4},
        # 'cuedend_contextsoma': {'data_path': data_path_cuedend_contextsoma, 'update_prior': update_prior_dict_cuedend_contextsoma, 'flow_idx': 4},
        # 'cuesoma_contextsoma_cellsoma': {'data_path': data_path_cuesoma_contextsoma_cellsoma, 'update_prior': update_prior_dict_cuesoma_contextsoma_cellsoma, 'flow_idx': 4},
        # 'cuedend_contextdend_celldend': {'data_path': data_path_cuedend_contextdend_celldend, 'update_prior': update_prior_dict_cuedend_contextdend_celldend, 'flow_idx': 4},
        'cuesoma_contextsoma_celldend': {'data_path': data_path_cuesoma_contextsoma_celldend, 'update_prior': update_prior_dict_cuesoma_contextsoma_celldend, 'flow_idx': 3},
    }

    for name, path_dict in res_paths.items():
        data_path = path_dict['data_path']
        update_prior_dict = path_dict['update_prior']
        
        res_dict = get_opt_data(data_path)
        flow_idx = path_dict['flow_idx']
        theta = res_dict['theta_list'][flow_idx]
        theta_idx = np.argmin(res_dict['error_list'][flow_idx])

        with open(f'{data_path}jaxley_net.pkl', 'rb') as f:
            net, gid_ranges = pickle.load(f)

        num_E_cells, num_I_cells = len(gid_ranges['E']), len(gid_ranges['I'])
        num_cue_cells = len(gid_ranges['cue'])

        params, _ = set_train_parameters(net, gid_ranges)
        prior_dict = get_prior_dict()
        update_prior_dict(prior_dict)

        input_list = jnp.array([[-2,-2,1], [2,2,1], [-2, 2,1], [2,-2,1],
                                [-2,-2,-1], [2,2,-1], [-2, 2,-1], [2,-2,-1]])
        num_cond = input_list.shape[0]
        input_data = [get_currents(input_list[idx], gid_ranges, t_max, dt) for idx in range(num_cond)]
        cue_currents = jnp.stack([input_data[idx][0] for idx in range(num_cond)])
        context_currents = jnp.stack([input_data[idx][1] for idx in range(num_cond)])
        targets = np.concatenate([input_data[idx][2][:2, ::downsample_factor] for idx in range(num_cond)], axis=1).T

        batch_size = 1
        cue_currents_batch = jnp.tile(cue_currents, (batch_size, 1, 1))
        context_currents_batch = jnp.tile(context_currents, (batch_size, 1, 1))
        print(cue_currents_batch.shape)

        jitted_simulate = jit(simulate_sweep)
        jitted_vmapped_simulate = vmap(jitted_simulate, in_axes=(0, None, 0, 0))

        # Run simulations in batch
        num_random_init = 20
        output_list = list()
        for start_idx in range(num_random_init):
            print(f'Batch: {start_idx}')
            theta_batch = theta[theta_idx:theta_idx+1, :]
            theta_batch = jnp.repeat(theta_batch, num_cond, axis=0)

            output = np.array(jitted_vmapped_simulate(theta_batch, params, cue_currents_batch, context_currents_batch))
            output = output[:, :, ::downsample_factor]
            output_list.append(output)
        output_array = np.concatenate(output_list)

        random_init_dict = {
            'name': name,
            'output_array': output_array,
            'targets': targets,
            'input_list': input_list,
            'theta': theta,
            'gid_ranges': gid_ranges
        }

        random_init_save_path = '/users/ntolley/data/ntolley/dendractor/random_initialization_memorycontext'
        fname = f'{name}_random_init.pkl'
        with open(f'{random_init_save_path}/{fname}', 'wb') as f:
            pickle.dump(random_init_dict,f)

