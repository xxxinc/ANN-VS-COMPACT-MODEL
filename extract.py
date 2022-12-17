from opt_para import opt_para
import numpy as np
import params
import matplotlib.pyplot as plt
from pvs_model_n import pvs_model_n
import os
from plotcurve import plotcurve
def extract(dir, index0, vg, vd, ids):
    # initial guess of fitting para
    delta = params.delta
    vxo = params.vxo
    wrs = params.WRs
    mu = params.mu
    ss = params.ss
    vt0 = params.vt0 

    para_init = [delta, vxo, wrs, mu, ss, vt0]

    para_op = opt_para(para_init, dir, index0)
    
    dir0 = os.path.join(dir, index0) # 地址

    #plotcurve.plotcurve(0, dir0, para_op)

    # 为了方便对于vs model的使用，我们在这一步对每种器件结构（每个index）
    # 的需要用到的vs model的结果进行计算，包括
    # 每个vd vg 对应的 I0, Gd0, Gm0

    if vg.any() and vd.any() and ids.any():
        id0 = pvs_model_n(np.array([vg, vd]), *para_op)
        dv = 0.05 # 差分求解电导、跨导的小量, 与数据保持一致
        vgdv = vg + dv
        iddvg = pvs_model_n(np.array([vgdv, vd]), *para_op)
        gm0 = (iddvg - id0)/(dv) #差分求跨导
        vddv = vd + dv
        iddvd = pvs_model_n(np.array([vg, vddv]), *para_op)
        gd0 = (iddvd - id0)/(dv)
        return [id0, gm0, gd0]
    
    else:
        return 0




if __name__ == '__main__':
    dir = r'C:\Users\xxxin\Documents\GitHub\DataBase\EDA2022\GAA_data'
    index0 = '1.txt'
    extract(dir, index0, [], [], [])