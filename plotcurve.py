import matplotlib.pyplot as plt
import numpy as np
import pvs_model
def plotcurve(flag, data1, data2):
    '''
    flag == 0: plot the pvs model and TCAD data comparation
               data1: path of the TCAD data
               data2: parameters of the pvs model
    flag == 1: plot the ANN model and TCAD data comparation
               data1: path of the TCAD data
               data2: ANN data
    flag == 2: plot the pvs model and ANN data comparation
               data1: parameter of the pvs model
               data2: ANN data
    '''
    if flag == 0:
        pathtacd = data1
        filename = pathtacd.split('\\')
        index = filename[-1][:-4]
        paras = data2

        with open(pathtacd, encoding='utf-8') as file:
            content = file.readlines()
        #print(content)
        content.pop(0)
        ivlist = []
        for line in content:
            dot = line.split()
            ivlist.append([float(i) for i in dot])
        #print(len(ivlist))
        ivdata = ivlist.copy()
        # plot output curve
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for i in range(15):
            vg = []
            vd = []
            id = []
            for j in range(15):
                dot = ivlist.pop(0)
                if(dot[1]==0): continue
                vg.append(dot[0])
                vd.append(dot[1])
                id.append(dot[2])
            label_tcad = str(vg[0]) + '-TCAD'
            label_pre = str(vg[0]) + '-PVS'
            id_pre = pvs_model.pvs_model(np.array([vg,vd]), *paras)

            ax1.plot(vd, id, label = label_tcad)
            ax1.plot(vd, pow(10, id_pre),'.',label = label_pre)
        plt.xlabel('Vg(V)',fontsize = 14)
        plt.ylabel('Id(A)',fontsize = 14)
        plt.title('Output Id-Vd Curve', fontsize = 14)
        #plt.legend(fontsize = 9)
        fig1.savefig('figures\\idvd-' + index)
        plt.close()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        for i in range(1,15):
            vg = []
            vd = []
            id = []
            for j in range(15):
                dot = ivdata[15 * j + i]
                vg.append(dot[0])
                vd.append(dot[1])
                id.append(dot[2])
            label_tcad = str(vg[0]) + '-TCAD'
            label_pre = str(vg[0]) + '-PVS'
            id_pre = pvs_model.pvs_model(np.array([vg,vd]), *paras)
            ax2.semilogy(vg, id, label = label_tcad)
            ax2.semilogy(vg, pow(10,id_pre), '.', label = label_pre)
        plt.xlabel('Vg(V)',fontsize = 14)
        plt.ylabel('Id(A)',fontsize = 14)
        plt.title('Transfer Id-Vg Curve', fontsize = 14)
        #plt.legend(fontsize = 9)
        fig2.savefig('figures\\idvg-' + index)
        plt.close()
