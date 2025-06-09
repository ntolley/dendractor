g_on = (-2, 1)
p_on = (0, 0.6)

g_off = (-20, -20)
p_off = (0, 0)

noise_g_on = (-1, -1)
noise_p_on = (1, 1)

cue_g_on = (-3, 1)
cue_p_on = (0, 1)

# Defaults for all priors
#_________________________________________________________________
def set_prior_noise(prior_dict):
    prior_dict['noise_Esoma_ampa_gS']['bounds'] = noise_g_on
    prior_dict['noise_I_ampa_gS']['bounds'] = noise_g_on

    prior_dict['noise_Esoma_ampa_pconn']['bounds'] = noise_p_on
    prior_dict['noise_I_ampa_pconn']['bounds'] = noise_p_on

def set_all_conn_off(prior_dict):
    # Cue
    prior_dict['cue_Esoma_ampa_gS']['bounds'] = g_off
    prior_dict['cue_Esoma_ampa_pconn']['bounds'] = p_off

    prior_dict['cue_Esoma_nmda_gS']['bounds'] = g_off
    prior_dict['cue_Esoma_nmda_pconn']['bounds'] = p_off

    prior_dict['cue_Edend_ampa_gS']['bounds'] = g_off
    prior_dict['cue_Edend_ampa_pconn']['bounds'] = p_off

    prior_dict['cue_Edend_nmda_gS']['bounds'] = g_off
    prior_dict['cue_Edend_nmda_pconn']['bounds'] = p_off

    # EEsoma
    prior_dict['EE_ampa_gS']['bounds'] = g_off
    prior_dict['EE_ampa_pconn']['bounds'] = p_off
    prior_dict['EE_nmda_gS']['bounds'] = g_off
    prior_dict['EE_nmda_pconn']['bounds'] = p_off

    # EEdend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_off
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_off
    prior_dict['EE_dend_nmda_gS']['bounds'] = g_off
    prior_dict['EE_dend_nmda_pconn']['bounds'] = p_off

    # IEsoma
    prior_dict['IE_gaba_gS']['bounds'] = g_off
    prior_dict['IE_gaba_pconn']['bounds'] =  p_off

    # IEdend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_off
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_off

    # II
    prior_dict['II_gaba_gS']['bounds'] = g_off
    prior_dict['II_gaba_pconn']['bounds'] =  p_off

    # EI
    prior_dict['EI_ampa_gS']['bounds'] = g_off
    prior_dict['EI_ampa_pconn']['bounds'] =  p_off

def set_prior_inhibitory_network(prior_dict):
    # IEsoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # IEdend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on

    # II
    prior_dict['II_gaba_gS']['bounds'] = g_on
    prior_dict['II_gaba_pconn']['bounds'] =  p_on

    # EI
    prior_dict['EI_ampa_gS']['bounds'] = g_on
    prior_dict['EI_ampa_pconn']['bounds'] =  p_on

def initalize_prior_dict(prior_dict):
    set_all_conn_off(prior_dict)
    set_prior_noise(prior_dict)
    set_prior_inhibitory_network(prior_dict)

#____________________________________________________________________

# Different variants of cue prior
def set_prior_cuesomaampa(prior_dict):
    prior_dict['cue_Esoma_ampa_gS']['bounds'] = cue_g_on
    prior_dict['cue_Esoma_ampa_pconn']['bounds'] = cue_p_on

def set_prior_cuesomanmda(prior_dict):
    prior_dict['cue_Esoma_nmda_gS']['bounds'] = cue_g_on
    prior_dict['cue_Esoma_nmda_pconn']['bounds'] = cue_p_on

def set_prior_cuedendampa(prior_dict):
    prior_dict['cue_Edend_ampa_gS']['bounds'] = cue_g_on
    prior_dict['cue_Edend_ampa_pconn']['bounds'] = cue_p_on

def set_prior_cuedendnmda(prior_dict):
    prior_dict['cue_Edend_nmda_gS']['bounds'] = cue_g_on
    prior_dict['cue_Edend_nmda_pconn']['bounds'] = cue_p_on
    
# Different variants of intrinsic conn prior

def set_prior_Esomaampa_Edendnmda(prior_dict):
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    prior_dict['EE_dend_nmda_gS']['bounds'] = g_on
    prior_dict['EE_dend_nmda_pconn']['bounds'] = p_on

def set_prior_Esomanmda_Edendampa(prior_dict):
    prior_dict['EE_nmda_gS']['bounds'] = g_on
    prior_dict['EE_nmda_pconn']['bounds'] = p_on

    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

#________________________________________________________________


def update_prior_dict_cuesomaampa_Esomaampa_Edendnmda(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuesomaampa(prior_dict)
    set_prior_Esomaampa_Edendnmda(prior_dict)

def update_prior_dict_cuesomaampa_Esomanmda_Edendampa(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuesomaampa(prior_dict)
    set_prior_Esomanmda_Edendampa(prior_dict)

def update_prior_dict_cuesomanmda_Esomaampa_Edendnmda(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuesomanmda(prior_dict)
    set_prior_Esomaampa_Edendnmda(prior_dict)

def update_prior_dict_cuesomanmda_Esomanmda_Edendampa(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuesomanmda(prior_dict)
    set_prior_Esomanmda_Edendampa(prior_dict)

def update_prior_dict_cuedendampa_Esomaampa_Edendnmda(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuedendampa(prior_dict)
    set_prior_Esomaampa_Edendnmda(prior_dict)

def update_prior_dict_cuedendampa_Esomanmda_Edendampa(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuedendampa(prior_dict)
    set_prior_Esomanmda_Edendampa(prior_dict)

def update_prior_dict_cuedendnmda_Esomaampa_Edendnmda(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuedendnmda(prior_dict)
    set_prior_Esomaampa_Edendnmda(prior_dict)

def update_prior_dict_cuedendnmda_Esomanmda_Edendampa(prior_dict):
    initalize_prior_dict(prior_dict)
    set_prior_cuedendnmda(prior_dict)
    set_prior_Esomanmda_Edendampa(prior_dict)








