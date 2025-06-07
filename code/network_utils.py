import sys
import jax
# from jax import config
jax.config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")
jax.config.update("jax_platform_name", "gpu")


import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad

import jaxley as jx
from jaxley.channels import Na, K, Leak, Km, CaL, CaT
from jaxley.synapses import IonotropicSynapse
from jaxley_synapses import AMPA, GABAa, GABAb, NMDA
import jaxley.optimize.transforms as jt
from jaxley.connect import fully_connect, connect, sparse_connect, connectivity_matrix_connect
import optax
import pickle

from jaxley.synapses.synapse import Synapse
from typing import Dict, Optional, Tuple
from jaxley.solver_gate import save_exp
from networkx import adjacency_matrix, gaussian_random_partition_graph, watts_strogatz_graph
from sklearn.linear_model import Ridge


class StimSynapse(Synapse):
    """
    Compute synaptic current and update synapse state for a generic ionotropic synapse.

    The synapse state "s" is the probability that a postsynaptic receptor channel is
    open, and this depends on the amount of neurotransmitter released, which is in turn
    dependent on the presynaptic voltage.

    The synaptic parameters are:
        - gS: the maximal conductance across the postsynaptic membrane (uS)
        - e_syn: the reversal potential across the postsynaptic membrane (mV)
        - k_minus: the rate constant of neurotransmitter unbinding from the postsynaptic
            receptor (s^-1)

    Details of this implementation can be found in the following book chapter:
        L. F. Abbott and E. Marder, "Modeling Small Networks," in Methods in Neuronal
        Modeling, C. Koch and I. Sergev, Eds. Cambridge: MIT Press, 1998.

    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,
            f"{prefix}_e_syn": 0.0,
            f"{prefix}_k_minus": 0.025,
        }
        self.synapse_states = {f"{prefix}_s": 0.2}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        """Return updated synapse state and current."""
        prefix = self._name
        v_th = 1.0  # mV
        delta = 10.0  # mV

        s_inf = 1.0 / (1.0 + save_exp((v_th - pre_voltage) / delta))
        tau_s = (1.0 - s_inf) / params[f"{prefix}_k_minus"]

        slope = -1.0 / tau_s
        exp_term = save_exp(slope * delta_t)
        new_s = states[f"{prefix}_s"] * exp_term + s_inf * (1.0 - exp_term)
        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        g_syn = params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn * (post_voltage - params[f"{prefix}_e_syn"])


def gaussian_tuning(tuned_val, state_val, sigma):
    return 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma.reshape(-1, 1)) * jnp.exp(-jnp.power((state_val - tuned_val) / sigma.reshape(-1, 1), 2.0))


def get_conn_matrix(src_indices, target_indices, seed=123, p_conn=1.0):
    graph = gaussian_random_partition_graph(n=600, s=10, v=1e10, p_in=1.0, p_out=0.0)
    # graph = watts_strogatz_graph(n=600, k=10, p=0.0)

    conn_rng = np.random.default_rng(seed)

    adj_matrix = adjacency_matrix(graph).toarray().astype(bool)
    adj_matrix = adj_matrix[src_indices, :]
    adj_matrix = adj_matrix[:, target_indices]
    mask = conn_rng.uniform(0, 1, size=(len(src_indices), len(target_indices))) < p_conn
    adj_matrix = adj_matrix * mask
    return adj_matrix

def linear_scale_forward(value, bounds, constrain_value=True):
    """Scale value in range (0,1) to range bounds"""
    return (bounds[0] + (value * (bounds[1] - bounds[0]))).astype(float)

def linear_scale_array(value, bounds, constrain_value=True):
    """Scale columns of array according to bounds"""
    return np.vstack(
        [linear_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def log_scale_forward(value, bounds, constrain_value=True):
    """log scale value in range (0,1) to range bounds in base 10"""
    rescaled_value = linear_scale_forward(value, bounds, constrain_value)
    return 10**rescaled_value

def log_scale_array(value, bounds, constrain_value=True):
    """log scale columns of array according to bounds in base 10"""
    return np.vstack(
        [log_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def make_network():
    comp = jx.Compartment()
    soma = jx.Branch(comp, ncomp=4)
    branch =  jx.Branch(comp, ncomp=4)
    
    E_cell = jx.Cell([soma, branch, branch, branch], parents=[-1, 0, 1, 2])

    E_cell.compute_xyz()
    E_cell.branch(0).set('length', 25.0)
    E_cell.branch(0).set('radius', 25.0 / 2)

    E_cell.branch(1).set('length', 100.0)
    E_cell.branch(1).set('radius', 2.5 / 2)

    E_cell.branch(2).set('length', 100.0)
    E_cell.branch(2).set('radius', 1.0 / 2)

    E_cell.branch(3).set('length', 100.0)
    E_cell.branch(3).set('radius', 0.5 / 2)

    E_cell.set('axial_resistivity', 300)

    I_cell = jx.Cell(soma, parents=[-1])
    I_cell.branch(0).set('length', 5.0)
    I_cell.branch(0).set('radius', 5.0 / 2)

    context_cell = jx.Cell(soma, parents=[-1])
    cue_cell = jx.Cell(soma, parents=[-1])
    rate_cell = jx.Cell(soma, parents=[-1])
    for cell in [context_cell, cue_cell, rate_cell]:
        cell.branch(0).set('length', 5.0)
        cell.branch(0).set('radius', 5.0 / 2)

    # num_E_cells = 100
    # num_I_cells = 50
    # num_context_cells = 50
    # num_cue_cells = 50
    num_E_cells = 50
    num_I_cells = 25
    num_context_cells = 25
    num_cue_cells = 25

    net_dict = {
        'E': {'num_cells': num_E_cells, 'cell': E_cell},
        'I': {'num_cells': num_I_cells, 'cell': I_cell},
        'context': {'num_cells': num_context_cells, 'cell': context_cell},
        'cue': {'num_cells': num_cue_cells, 'cell': cue_cell},
        'E_rate': {'num_cells': num_E_cells, 'cell': rate_cell},
        'I_rate': {'num_cells': num_I_cells, 'cell': rate_cell},
    }

    gid_ranges = dict()
    cell_list = list()
    cell_count = 0
    for name, cell_dict in net_dict.items():
        num_cells = cell_dict['num_cells']
        gid_ranges[name] = range(cell_count, cell_count + num_cells)
        cell_list.extend([cell_dict['cell'] for _ in range(num_cells)])
        cell_count += num_cells

    net = jx.Network(cell_list)

    for name in ['context', 'cue']:
        net.cell(gid_ranges[name]).insert(Na())
        net.cell(gid_ranges[name]).insert(K())
        net.cell(gid_ranges[name]).insert(Leak())
    
    e_km, e_dend_km, i_km = Km(), Km(), Km()
    e_km.change_name('E_Km')
    e_dend_km.change_name('E_dend_Km')
    i_km.change_name('I_Km')

    e_cat, e_dend_cat, i_cat = CaT(), CaT(), CaT()
    e_cat.change_name('E_CaT')
    e_dend_cat.change_name('E_dend_CaT')
    i_cat.change_name('I_CaT')

    e_cal, e_dend_cal, i_cal = CaL(), CaL(), CaL()
    e_cal.change_name('E_CaL')
    e_dend_cal.change_name('E_dend_CaL')
    i_cal.change_name('I_CaL')

    e_leak, e_dend_leak, i_leak = Leak(), Leak(), Leak()
    e_leak.change_name('E_Leak')
    e_dend_leak.change_name('E_dend_Leak')
    i_leak.change_name('I_Leak')

    e_na, e_dend_na, i_na = Na(), Na(), Na()
    e_na.change_name('E_Na')
    e_dend_na.change_name('E_dend_Na')
    i_na.change_name('I_Na')

    e_k, e_dend_k, i_k = K(), K(), K()
    e_k.change_name('E_K')
    e_dend_k.change_name('E_dend_K')
    i_k.change_name('I_K')


    net.cell(gid_ranges['E']).branch(0).insert(e_leak)
    net.cell(gid_ranges['E']).branch(0).insert(e_na)
    net.cell(gid_ranges['E']).branch(0).insert(e_k)
    net.cell(gid_ranges['E']).branch(0).insert(e_km)
    net.cell(gid_ranges['E']).branch(0).insert(e_cat)
    net.cell(gid_ranges['E']).branch(0).insert(e_cal)

    net.cell(gid_ranges['E']).branch([1,2,3]).insert(e_dend_leak)
    net.cell(gid_ranges['E']).branch([1,2,3]).insert(e_dend_na)
    net.cell(gid_ranges['E']).branch([1,2,3]).insert(e_dend_k)
    net.cell(gid_ranges['E']).branch([1,2,3]).insert(e_dend_km)
    net.cell(gid_ranges['E']).branch([1,2,3]).insert(e_dend_cat)
    net.cell(gid_ranges['E']).branch([1,2,3]).insert(e_dend_cal)

    net.cell(gid_ranges['I']).insert(i_leak)
    net.cell(gid_ranges['I']).insert(i_na)
    net.cell(gid_ranges['I']).insert(i_k)
    net.cell(gid_ranges['I']).insert(i_km)
    net.cell(gid_ranges['I']).insert(i_cat)
    net.cell(gid_ranges['I']).insert(i_cal)

    for name in ['E_rate', 'I_rate']:
        net.cell(gid_ranges[name]).insert(Leak())
        net.cell(gid_ranges[name]).set('Leak_eLeak', 0.0)
        net.cell(gid_ranges[name]).set('v', 0.0)

    cue_Esoma_ampa_synapse = AMPA()
    # cue_Esoma_ampa_synapse = NMDA()
    cue_Esoma_ampa_synapse.change_name('cue_Esoma_ampa')

    # cue_Edend_ampa_synapse = AMPA()
    cue_Edend_ampa_synapse = NMDA()
    cue_Edend_ampa_synapse.change_name('cue_Edend_ampa')

    context_Esoma_ampa_synapse = AMPA()
    # context_Esoma_ampa_synapse = NMDA()
    context_Esoma_ampa_synapse.change_name('context_Esoma_ampa')

    # context_Edend_ampa_synapse = AMPA()
    context_Edend_ampa_synapse = NMDA()
    context_Edend_ampa_synapse.change_name('context_Edend_ampa')

    cue_I_ampa_synapse = AMPA()
    cue_I_ampa_synapse.change_name('cue_I_ampa')

    context_I_ampa_synapse = AMPA()
    context_I_ampa_synapse.change_name('context_I_ampa')

    EI_ampa_synapse = AMPA()
    EI_ampa_synapse.change_name('EI_ampa')

    EE_ampa_synapse = AMPA()
    EE_ampa_synapse.change_name('EE_ampa')

    # EE_dend_ampa_synapse = AMPA()
    EE_dend_ampa_synapse = NMDA()
    EE_dend_ampa_synapse.change_name('EE_dend_ampa')

    IE_gaba_synapse = GABAa()
    IE_gaba_synapse.change_name('IE_gaba')

    IE_dend_gaba_synapse = GABAa()
    IE_dend_gaba_synapse.change_name('IE_dend_gaba')

    II_gaba_synapse = GABAa()
    II_gaba_synapse.change_name('II_gaba')

    exp_synapse = StimSynapse()
    exp_synapse.change_name('exp_synapse')

    #  ******* CLUSTERED CONNECTIVITY *********
    E_indices, I_indices = np.arange(num_E_cells), np.arange(num_I_cells) * (num_E_cells // num_I_cells)
    # E->I soma
    connectivity_matrix_connect(net.cell(gid_ranges['E']).branch(0).comp(0), net.cell(gid_ranges['I']).branch(0).comp(0), synapse_type=EI_ampa_synapse,
                                connectivity_matrix=get_conn_matrix(E_indices, I_indices, seed=123))

    # E->E soma and dendrite
    connectivity_matrix_connect(net.cell(gid_ranges['E']).branch(0).comp(0), net.cell(gid_ranges['E']).branch(0).comp(0), synapse_type=EE_ampa_synapse, 
                                connectivity_matrix=get_conn_matrix(E_indices, E_indices, seed=124))
    connectivity_matrix_connect(net.cell(gid_ranges['E']).branch(0).comp(0), net.cell(gid_ranges['E']).branch(3).comp(3), synapse_type=EE_dend_ampa_synapse, 
                                connectivity_matrix=get_conn_matrix(E_indices, E_indices, seed=224))

    # I->E soma and dendrite
    connectivity_matrix_connect(net.cell(gid_ranges['I']).branch(0).comp(0), net.cell(gid_ranges['E']).branch(0).comp(0), synapse_type=IE_gaba_synapse,
                                connectivity_matrix=get_conn_matrix(I_indices, E_indices, seed=125))
    connectivity_matrix_connect(net.cell(gid_ranges['I']).branch(0).comp(0), net.cell(gid_ranges['E']).branch(3).comp(3), synapse_type=IE_dend_gaba_synapse,
                                connectivity_matrix=get_conn_matrix(I_indices, E_indices, seed=225))


    connectivity_matrix_connect(net.cell(gid_ranges['I']).branch(0).comp(0), net.cell(gid_ranges['I']).branch(0).comp(0), synapse_type=II_gaba_synapse,
                                         connectivity_matrix=get_conn_matrix(I_indices, I_indices, seed=126))

    # Non overlapping cells for input/output of network
    E_in_gids = list(gid_ranges['E'])[::2]
    E_out_gids = list(gid_ranges['E'])[1::2]
    E_out_rate_gids = list(gid_ranges['E_rate'])[1::2]

    I_in_gids = list(gid_ranges['I'])[::2]
    I_out_gids = list(gid_ranges['I'])[1::2]
    I_out_rate_gids = list(gid_ranges['I_rate'])[1::2]

    sparse_connect(net.cell(gid_ranges['cue']).branch(0).comp(0), net.cell(E_in_gids).branch(0).comp(0), synapse_type=cue_Esoma_ampa_synapse, p=0.1)
    sparse_connect(net.cell(gid_ranges['cue']).branch(0).comp(0), net.cell(E_in_gids).branch(3).comp(3), synapse_type=cue_Edend_ampa_synapse, p=0.1)
    sparse_connect(net.cell(gid_ranges['cue']).branch(0).comp(0), net.cell(I_in_gids).branch(0).comp(0), synapse_type=cue_I_ampa_synapse, p=0.1)

    sparse_connect(net.cell(gid_ranges['context']).branch(0).comp(0), net.cell(E_in_gids).branch(0).comp(0), synapse_type=context_Esoma_ampa_synapse, p=0.1)
    sparse_connect(net.cell(gid_ranges['context']).branch(0).comp(0), net.cell(E_in_gids).branch(3).comp(3), synapse_type=context_Edend_ampa_synapse, p=0.1)
    sparse_connect(net.cell(gid_ranges['context']).branch(0).comp(0), net.cell(I_in_gids).branch(0).comp(0), synapse_type=context_I_ampa_synapse, p=0.1)



    connectivity_matrix_connect(net.cell(E_out_gids).branch(0).comp(0), net.cell(E_out_rate_gids).branch(0).comp(0),
                                synapse_type=exp_synapse, connectivity_matrix=np.eye(len(E_out_gids)).astype(bool))
    connectivity_matrix_connect(net.cell(I_out_gids).branch(0).comp(0), net.cell(I_out_rate_gids).branch(0).comp(0),
                                synapse_type=exp_synapse, connectivity_matrix=np.eye(len(I_out_gids)).astype(bool))

    net.copy_node_property_to_edges("global_cell_index")

    return net, gid_ranges

def set_train_parameters(net, gid_ranges):
    net.set('exp_synapse_e_syn', 10.0)
    net.set('exp_synapse_k_minus', 0.1)

    net.set('E_Km_gKm', 1e-5)
    net.set('E_CaL_gCaL', 1e-3)
    net.set('E_CaT_gCaT', 1e-3)
    net.set('I_Km_gKm', 1e-5)
    net.set('I_CaL_gCaL', 1e-3)
    net.set('I_CaT_gCaT', 1e-3)

    net.delete_trainables()
    # # Define which parameters to train.
    net.select(edges="all").make_trainable("IE_gaba_gS")
    net.select(edges="all").make_trainable("II_gaba_gS")
    net.select(edges="all").make_trainable("EI_ampa_gS")
    net.select(edges="all").make_trainable("EE_ampa_gS")
    net.select(edges="all").make_trainable("EE_dend_ampa_gS")
    net.select(edges="all").make_trainable("IE_dend_gaba_gS")

    net.select(edges="all").make_trainable('cue_Esoma_ampa_gS')
    net.select(edges="all").make_trainable('cue_Edend_ampa_gS')
    net.select(edges="all").make_trainable('context_Esoma_ampa_gS')
    net.select(edges="all").make_trainable('context_Edend_ampa_gS')
    net.select(edges="all").make_trainable('cue_I_ampa_gS')
    net.select(edges="all").make_trainable('context_I_ampa_gS')

    net.make_trainable("E_Leak_gLeak")
    net.make_trainable("E_dend_Leak_gLeak")
    net.make_trainable("I_Leak_gLeak")

    net.make_trainable("E_Km_gKm")
    net.make_trainable("E_CaL_gCaL")
    net.make_trainable("E_CaT_gCaT")

    net.make_trainable("E_dend_Km_gKm")
    net.make_trainable("E_dend_CaL_gCaL")
    net.make_trainable("E_dend_CaT_gCaT")

    net.make_trainable("I_Km_gKm")
    net.make_trainable("I_CaL_gCaL")
    net.make_trainable("I_CaT_gCaT")

    parameters = net.get_parameters()
    key = jax.random.PRNGKey(0)
    parameters.append({'W_out': jax.random.uniform(key=key,shape=(2, len(gid_ranges['E']),), minval=-0.5, maxval=0.5)})
    parameters.append({'b_out': jax.random.uniform(key=key,shape=(2, 1), minval=-0.1, maxval=0.1)})

    return parameters, None

def get_parameter_names():
    conn_names = ["cue_Esoma_ampa", "context_Esoma_ampa", "cue_Edend_ampa", "context_Edend_ampa",
                  "cue_I_ampa", "context_I_ampa",
                  "IE_gaba", "II_gaba", "EI_ampa", "EE_ampa", "IE_dend_gaba", "EE_dend_ampa"]

    conn_g_names = [f'{name}_gS' for name in conn_names]
    conn_pconn_names = [f'{name}_pconn' for name in conn_names]

    biophysics_names = ["E_Leak_gLeak", "E_dend_Leak_gLeak", "I_Leak_gLeak",
                       "E_Km_gKm", "E_CaL_gCaL", "E_CaT_gCaT",
                       "I_Km_gKm", "I_CaL_gCaL", "I_CaT_gCaT",
                       "E_dend_Km_gKm", "E_dend_CaL_gCaL", "E_dend_CaT_gCaT"]

    key_order = conn_g_names + conn_pconn_names + biophysics_names

    return key_order, conn_names, biophysics_names

def simulate(params, cue_currents, context_currents):
    net.delete_stimuli()
    
    data_stimuli = net.cell(list(gid_ranges['cue'])).branch(0).comp(0).data_stimulate(cue_currents)
    data_stimuli = net.cell(list(gid_ranges['context'])).branch(0).comp(0).data_stimulate(context_currents, data_stimuli=data_stimuli)

    net.delete_recordings()
    net.branch(0).comp(0).record('v')
    s = jx.integrate(net, t_max=t_max, params=params, checkpoint_lengths=checkpoints, data_stimuli=data_stimuli)
    # s = jx.integrate(net, t_max=t_max, params=params, data_stimuli=data_stimuli)

    return s

def get_prior_dict():
    # Default is cue->soma, context->dend, cell->soma+dend
    prior_dict = {
        "cue_Esoma_ampa_gS": {'bounds': (-3, -3), 'rescale_function': log_scale_forward},
        "context_Esoma_ampa_gS": {'bounds': (-3, -3), 'rescale_function': log_scale_forward},
        "cue_I_ampa_gS": {'bounds': (-3, -3), 'rescale_function': log_scale_forward},
        "context_I_ampa_gS": {'bounds': (-3, -3), 'rescale_function': log_scale_forward},
        "IE_gaba_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "II_gaba_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "EI_ampa_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "EE_ampa_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},

        "cue_Edend_ampa_gS": {'bounds': (-20, -20), 'rescale_function': log_scale_forward},
        "context_Edend_ampa_gS": {'bounds': (-20, -20), 'rescale_function': log_scale_forward},
        "IE_dend_gaba_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        "EE_dend_ampa_gS": {'bounds': (-9, -2), 'rescale_function': log_scale_forward},
        
        "cue_Esoma_ampa_pconn": {'bounds': (1.0, 1.0), 'rescale_function': linear_scale_forward},
        "context_Esoma_ampa_pconn": {'bounds': (1.0, 1.0), 'rescale_function': linear_scale_forward},
        "cue_I_ampa_pconn": {'bounds': (1.0, 1.0), 'rescale_function': linear_scale_forward},
        "context_I_ampa_pconn": {'bounds': (1.0, 1.0), 'rescale_function': linear_scale_forward},
        "IE_gaba_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "II_gaba_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "EI_ampa_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "EE_ampa_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},

        "cue_Edend_ampa_pconn": {'bounds': (0.0, 0.0), 'rescale_function': linear_scale_forward},
        "context_Edend_ampa_pconn": {'bounds': (0.0, 0.0), 'rescale_function': linear_scale_forward},
        "IE_dend_gaba_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},
        "EE_dend_ampa_pconn": {'bounds': (0, 0.3), 'rescale_function': linear_scale_forward},

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

def loss_fn(opt_params, cue_currents, context_currents, target):
    rate_scale = 1.0
    params = transform.forward(opt_params)
    s = simulate(params, cue_currents, context_currents)

    output = jnp.matmul(params[-2]['W_out'], s[list(gid_ranges['E_rate']), :] * rate_scale).squeeze() + params[-1]['b_out']

    # Only apply loss to first two dimensions for target location, not context in third dimension
    loss = jnp.mean(jnp.square(output[:2, burn_in:] - target[:2,burn_in:]))

    return loss

def batched_loss_fn(opt_params, cue_currents, context_currents, target):
    loss = batched_loss(opt_params, cue_currents, context_currents, target)
    return jnp.mean(loss)
    

def get_currents(inputs, gid_ranges, t_max=500, dt=0.025):
    time_vec = jnp.arange(0, t_max, dt)
    
    cue_rng = np.random.default_rng(12345)

    if inputs[2] == 1:
        context_rng = np.random.default_rng(111)
    elif inputs[2] == -1:
        context_rng = np.random.default_rng(222)

    # calculate cue amplitudes
    cue_dim = 2
    cue_tuning = cue_rng.uniform(-3, 3, (len(gid_ranges['cue']), cue_dim))
    cue_sigma = np.array(0.3)
    cue_tuning_denom = gaussian_tuning(0, 0, cue_sigma) * cue_dim # ensures input intensity equals 1

    cue_amplitudes = gaussian_tuning(inputs[0:2], cue_tuning, cue_sigma)
    cue_amplitudes = (np.sum(cue_amplitudes, axis=1) / cue_tuning_denom).reshape(-1, 1)
    baseline_amplitudes = gaussian_tuning(np.array([0.0, 0.0]), cue_tuning, cue_sigma)
    baseline_amplitudes = (np.sum(baseline_amplitudes, axis=1) / cue_tuning_denom).reshape(-1, 1)

    # calculate context amplitudes
    context_dim = 1
    context_tuning = context_rng.uniform(-3, 3, (len(gid_ranges['context']), context_dim))
    context_sigma = np.array(0.3)
    context_tuning_denom = gaussian_tuning(0, 0, context_sigma) * context_dim

    context_amplitudes = gaussian_tuning(np.array([0.0]), context_tuning, context_sigma)
    context_amplitudes = (np.sum(context_amplitudes, axis=1) / context_tuning_denom).reshape(-1, 1)

    # Define input start/stop times
    stim_len = 1000
    stim_scaling = 0.1

    cue_start = 28000
    cue_stop = cue_start + stim_len

    context_start = 10000
    context_stop = context_start + stim_len

    # cue currents
    cue_currents = np.zeros((len(gid_ranges['cue']), len(time_vec)))
    cue_currents[:, cue_start:cue_stop] = cue_amplitudes

    cue_currents = jnp.asarray(cue_currents * stim_scaling)

    # context currents
    context_currents = np.zeros((len(gid_ranges['context']), len(time_vec)))
    context_currents[:, context_start:context_stop] = context_amplitudes

    context_currents = jnp.asarray(context_currents * stim_scaling)

    target = jnp.zeros((3, len(time_vec) + 1))
    target = target.at[:2, cue_start:].set(inputs[0:2].reshape(-1,1) * inputs[2])

    return cue_currents, context_currents, target

def get_currents_nocontext(inputs, gid_ranges, t_max=500, dt=0.025):
    time_vec = jnp.arange(0, t_max, dt)
    
    cue_rng = np.random.default_rng(12345)

    # calculate cue amplitudes
    cue_dim = 2
    cue_tuning = cue_rng.uniform(-3, 3, (len(gid_ranges['cue']), cue_dim))
    cue_sigma = np.array(0.3)
    cue_tuning_denom = gaussian_tuning(0, 0, cue_sigma) * cue_dim # ensures input intensity equals 1

    cue_amplitudes = gaussian_tuning(inputs[0:2], cue_tuning, cue_sigma)
    cue_amplitudes = (np.sum(cue_amplitudes, axis=1) / cue_tuning_denom).reshape(-1, 1)
    baseline_amplitudes = gaussian_tuning(np.array([0.0, 0.0]), cue_tuning, cue_sigma)
    baseline_amplitudes = (np.sum(baseline_amplitudes, axis=1) / cue_tuning_denom).reshape(-1, 1)


    # Define input start/stop times
    stim_len = 1000
    stim_scaling = 0.1

    cue_start = 10000
    cue_stop = cue_start + stim_len

    context_start = 10000
    context_stop = context_start + stim_len

    # cue currents
    cue_currents = np.zeros((len(gid_ranges['cue']), len(time_vec)))
    cue_currents[:, cue_start:cue_stop] = cue_amplitudes

    cue_currents = jnp.asarray(cue_currents * stim_scaling)

    # context currents
    context_currents = np.zeros((len(gid_ranges['context']), len(time_vec)))

    target = jnp.zeros((3, len(time_vec) + 1))
    target = target.at[:2, cue_start:].set(inputs[0:2].reshape(-1,1) * inputs[2])

    return cue_currents, context_currents, target