g_on = (-9, -2)
p_on = (0, 0.3)

g_off = (-20, -20)
p_off = (0, 0)

def set_prior_cuesoma_contextsoma(prior_dict):
    prior_dict['cue_ampa_gS']['bounds'] = (-3, -3)
    prior_dict['context_ampa_gS']['bounds'] = (-3, -3)
    prior_dict['cue_ampa_pconn']['bounds'] = (1, 1)
    prior_dict['context_ampa_pconn']['bounds'] = (1, 1)

    prior_dict['cue_dend_ampa_gS']['bounds'] = g_off
    prior_dict['context_dend_ampa_gS']['bounds'] = g_off
    prior_dict['cue_dend_ampa_pconn']['bounds'] = p_off
    prior_dict['context_dend_ampa_pconn']['bounds'] = p_off

def update_prior_dict_Esoma_Isoma(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_off
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_off

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_off
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_off

def update_prior_dict_Edend_Idend(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_off
    prior_dict['EE_ampa_pconn']['bounds'] = p_off

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_off
    prior_dict['IE_gaba_pconn']['bounds'] = p_off

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on

def update_prior_dict_Esoma_Idend(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_off
    prior_dict['IE_gaba_pconn']['bounds'] =  p_of

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_off
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_off

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on

def update_prior_dict_Edend_Isoma(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_off
    prior_dict['EE_ampa_pconn']['bounds'] = p_off

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_off
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_off

def update_prior_dict_Esomadend_Isoma(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_off
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_off

def update_prior_dict_Esomadend_Idend(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_off
    prior_dict['IE_gaba_pconn']['bounds'] =  p_off

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on

def update_prior_dict_Esoma_Isomadend(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_off
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_off

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on

def update_prior_dict_Edend_Isomadend(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_off
    prior_dict['EE_ampa_pconn']['bounds'] = p_off

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on

def update_prior_dict_Esomadend_Isomadend(prior_dict):
    set_prior_cuesoma_contextsoma(prior_dict)

    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = g_on
    prior_dict['EE_ampa_pconn']['bounds'] = p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = p_on





