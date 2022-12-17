import numpy as np
from scipy.optimize import curve_fit
from pvs_model import pvs_model
import params
import os
import pandas as pd
def opt_para(para_init, dir, index0):
    
    dir0 = os.path.join(dir, index0)

    vg_data = []
    vd_data = []
    id_data = []

    index_ = index0[:-4]

    with open(dir0, encoding='utf-8') as file:
        content = file.readlines()
    #print(content)
    content.pop(0)
    for line in content:
        dot = np.array(line.split()).astype(np.float32) 
        #print(dot)
        if(dot[1] == 0):
            # 跳过vd = 0，由VS MODEL 决定的
            continue
        else:
            vg_data.append(dot[0])
            vd_data.append(dot[1])
            id_data.append(dot[2])

    # vspara_initial = [[0,0,0,0,0,0]]
    # df_vspara = pd.DataFrame(vspara_initial, index=[0], columns=['delta', 'vxo', 'wrs', 'mu', 'ss', 'vt0'])
    # df_vspara.to_csv('vs_params.csv', index_label='index')

    x_data = np.array([vg_data, vd_data])
    y_data = np.log10(np.array(id_data))
    #np.savetxt('data.txt',np.array([vg_data, vd_data, id_data]).T)
    upb = [0,   0.1, 1,   50,   0.059, 0.2]
    bob = [0.5, 10,  200, 1000, 0.1,   0.6]
    para_bounds=(upb, bob)
    popt, pcov = curve_fit(pvs_model, x_data, y_data, p0=para_init, bounds=para_bounds, method='trf')
    df_popt = pd.DataFrame([popt], index = [int(index_)])
    df_popt.to_csv('vs_params.csv', mode = 'a', header=False)

    return popt


if __name__ == '__main__':
    dir = r'C:\Users\xxxin\Documents\GitHub\DataBase\EDA2022\GAA_data'
    index0 = '1.txt'
    delta = params.delta
    vxo = params.vxo
    wrs = params.WRs
    mu = params.mu
    ss = params.ss
    vt0 = params.vt0 

    para_init = [delta, vxo, wrs, mu, ss, vt0]
    opt_para(para_init, dir, index0)

    