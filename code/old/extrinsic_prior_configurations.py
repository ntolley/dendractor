# recurrent_g_on = (-4, 1)
recurrent_g_on = (-2, 1)
recurrent_p_on = (0, 0.3)

g_off = (-20, -20)
p_off = (0, 0)

input_g_on = (-4, 1)
input_p_on = (0, 1)

def set_prior_Esomadend_Isomadend(prior_dict):
    # Esoma
    prior_dict['EE_ampa_gS']['bounds'] = recurrent_g_on
    prior_dict['EE_ampa_pconn']['bounds'] = recurrent_p_on

    # Isoma
    prior_dict['IE_gaba_gS']['bounds'] = recurrent_g_on
    prior_dict['IE_gaba_pconn']['bounds'] =  recurrent_p_on

    # Edend
    prior_dict['EE_dend_ampa_gS']['bounds'] = recurrent_g_on
    prior_dict['EE_dend_ampa_pconn']['bounds'] = recurrent_p_on

    # Idend
    prior_dict['IE_dend_gaba_gS']['bounds'] = recurrent_g_on
    prior_dict['IE_dend_gaba_pconn']['bounds'] = recurrent_p_on

def update_prior_dict_contextsoma_cuesoma(prior_dict):
    set_prior_Esomadend_Isomadend(prior_dict)

    prior_dict['cue_Esoma_ampa_gS']['bounds'] = input_g_on
    prior_dict['context_Esoma_ampa_gS']['bounds'] = input_g_on
    prior_dict['cue_Esoma_ampa_pconn']['bounds'] = input_p_on
    prior_dict['context_Esoma_ampa_pconn']['bounds'] = input_p_on

    prior_dict['cue_Edend_ampa_gS']['bounds'] = g_off
    prior_dict['context_Edend_ampa_gS']['bounds'] = g_off
    prior_dict['cue_Edend_ampa_pconn']['bounds'] = p_off
    prior_dict['context_Edend_ampa_pconn']['bounds'] = p_off

def update_prior_dict_contextdend_cuedend(prior_dict):
    set_prior_Esomadend_Isomadend(prior_dict)

    prior_dict['cue_Esoma_ampa_gS']['bounds'] = g_off
    prior_dict['context_Esoma_ampa_gS']['bounds'] = g_off
    prior_dict['cue_Esoma_ampa_pconn']['bounds'] = p_off
    prior_dict['context_Esoma_ampa_pconn']['bounds'] = p_off

    prior_dict['cue_Edend_ampa_gS']['bounds'] = input_g_on
    prior_dict['context_Edend_ampa_gS']['bounds'] = input_g_on
    prior_dict['cue_Edend_ampa_pconn']['bounds'] = input_p_on
    prior_dict['context_Edend_ampa_pconn']['bounds'] = input_p_on

def update_prior_dict_contextsoma_cuedend(prior_dict):
    set_prior_Esomadend_Isomadend(prior_dict)

    prior_dict['cue_Esoma_ampa_gS']['bounds'] = g_off
    prior_dict['context_Esoma_ampa_gS']['bounds'] = input_g_on
    prior_dict['cue_Esoma_ampa_pconn']['bounds'] = p_off
    prior_dict['context_Esoma_ampa_pconn']['bounds'] = input_p_on

    prior_dict['cue_Edend_ampa_gS']['bounds'] = input_g_on
    prior_dict['context_Edend_ampa_gS']['bounds'] = g_off
    prior_dict['cue_Edend_ampa_pconn']['bounds'] = input_p_on
    prior_dict['context_Edend_ampa_pconn']['bounds'] = p_off

def update_prior_dict_contextdend_cuesoma(prior_dict):
    set_prior_Esomadend_Isomadend(prior_dict)

    prior_dict['cue_Esoma_ampa_gS']['bounds'] = input_g_on
    prior_dict['context_Esoma_ampa_gS']['bounds'] = g_off
    prior_dict['cue_Esoma_ampa_pconn']['bounds'] = input_p_on
    prior_dict['context_Esoma_ampa_pconn']['bounds'] = p_off

    prior_dict['cue_Edend_ampa_gS']['bounds'] = g_off
    prior_dict['context_Edend_ampa_gS']['bounds'] = input_g_on
    prior_dict['cue_Edend_ampa_pconn']['bounds'] = input_p_on
    prior_dict['context_Edend_ampa_pconn']['bounds'] = p_off





