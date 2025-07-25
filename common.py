import numpy as np
import torch


STATE_COLS = ['gender', 'age', 'elixhauser', 're_admission',
       'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
       'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN',
       'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT',
       'SGPT', 'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count',
       'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE',
       'Arterial_lactate', 'HCO3', 'mechvent', 'Shock_Index', 'PaO2_FiO2',
       'SOFA', 'SIRS', 'cumulated_balance']


def get_states(df, latent=True):
    """
    Helper function. Given a df, returns a Tensor of all latent space representations 
    in that df. Pandas DataFrames store the string representation of arrays, so we have to 
    parse the contents of each cell.
    """
    if latent:
        states = df['latent_state']
        states = np.stack([np.fromstring(state.strip('[]'), sep=' ') for state in states])
        states = torch.from_numpy(states).to(dtype=torch.float32) # (264589, 20)

    else:
        pass # TODO 

    return states


def get_action_id(row):
    """
    Helper function. Given an IV bin and a vaso bin, return the unique action ID, in [0, 24].
    """
    return (5 * row['iv_bin']) + row['vaso_bin']


def get_action_tuple(action_id):
    """
    Helper function. Given an action ID, return the (iv, vaso) tuple.
    """
    iv = action_id // 5
    vaso = action_id % 5
    return (iv, vaso)