import os
import numpy as np

def get_parameter_bound(dir): #从config.txt文件中读取参数的上下界
    '''
        GET vdd, vdlin, parameters from config.txt
        Return vdd,vdlin, parameter name, ub and lb without preprocess the values
    '''
    eps = 1e-8 #防止lg0的出现

    f = open(dir, mode  = 'r', encoding = 'UTF-8') #r只读 
    lines = f.readlines() #readline() readlines()
    f.close()
    
    vdd = lines[0].strip()# 删除字符串中多余字符
    vdd = vdd.split(' ')
    vdd = float(vdd[1])
    
    vlin = lines[1].strip()
    vlin = vlin.split(' ')
    vlin = float(vlin[1])

    para_name = lines[2].strip()
    para_name = para_name.split(' ')
    para_name = para_name[1]
    para_name = para_name.split(',')
 
    lowerbound = lines[3].strip()
    lowerbound = lowerbound.split(' ')
    lowerbound = lowerbound[1]
    lowerbound = lowerbound.split(',')
    for i in range(len(lowerbound)):
        lowerbound[i] = float(lowerbound[i])
    
    upperbound = lines[4].strip()
    upperbound = upperbound.split(' ')
    upperbound = upperbound[1]
    upperbound = upperbound.split(',')
    for i in range(len(upperbound)):
        upperbound[i] = float(upperbound[i])
        
    lowerbound = np.array(lowerbound)
    upperbound = np.array(upperbound)
    lowerbound = np.log10(abs(lowerbound) + eps)
    upperbound = np.log10(abs(upperbound) + eps)
    return vdd, vlin, para_name, lowerbound, upperbound

def get_Ids(dir):
    '''
        从num.txt文件中读取不同电压下的电流值
        Return Vgs Vds Ids
    '''

    f = open(dir, mode  = 'r', encoding = 'UTF-8') #r只读 w写入 a追加模式
    text = f.readlines() #readline() readlines()
    f.close()

    Vg = []
    Vd = [] 
    Ids = []
    
    for i in range(1,len(text)):
        inter = text[i].strip() # 删除字符串中多余字符
        inter = inter.split('\t')
        Vg.append(float(inter[0]))
        Vd.append(float(inter[1]))
        Ids.append(float(inter[2]))
    Vg = np.array(Vg)
    Vd = np.array(Vd)
    Ids = np.array(np.abs(Ids))
    return Vg, Vd, Ids

def get_data(dir, trainortest): 
    '''
        从parameter.txt文件中提取完整的数据
        #dir为存放config.txt parametertrain.txt num.txt的目录, e.g. 'planar_data'
        #返回值para为器件参数+栅电压+漏电压，target为源漏电流
    '''
    path_config = os.path.join(dir, r"config.txt")
    #print(path_config)
    
    vdd, vlin, para_name, lowerbound, upperbound = get_parameter_bound(path_config)
    #print(path_config)
    #print(vdd, vlin, para_name, lowerbound, upperbound)
    #print(len(para_name))
    
    ### 得根据每种参数的值域来进行预处理
    
    if trainortest == 0:
        path_train = os.path.join(dir, r"parametertrain.txt")
    else:
        if trainortest == 1:
            path_train = os.path.join(dir, r"parametertest.txt")  
    f = open(path_train, mode  = 'r', encoding = 'UTF-8') #r只读 w写入 a追加模式
    text = f.readlines() #readline() readlines()
    f.close()
    #print(text)
    para, target, Gm, Gd = [], [], [], []
    
    for i in range(1,len(text)):
        inter = text[i].strip() # 删除字符串中多余字符
        inter = inter.strip(' ')
        inter = inter.split(',')
        #print(inter)
        parameter0 =[]
        for j in range(len(para_name)):
            parameter0.append(float(inter[j]))
        index = str(int(inter[len(para_name)]))+'.txt'
        path_Ids = os.path.join(dir, index)

        # 获得Vg Vd Ids 的numpy array
        Vg, Vd, Ids = get_Ids(path_Ids)
        Gm0 = []
        Gd0 = []

        # 由于现在是对一个个器件获取参数，所以这些Vg Vd Ids都是关于一个器件的，可以求相应的跨导电导
        # 我们所求的的导数是输出(logI)对应Vgs\Vds的
        dv = 0.05 #考虑到序列可以是Vg不变，变化Vd 或者相反，所以取一个不为0的dv
        numdvd = int(int(vdd*100) / int(dv*100) + 1) # e.g. 0.7 / 0.05 = 14 + 1
        len_of_data = Vg.size #获得数据的大小，利用循环来求解
        #print(len_of_data)
        numdvg = int(len_of_data / (numdvd))
        #print('vdd', vdd)
        #print('numdvd:',numdvd)
        #print('numdvg:',numdvg)

        for vvg in range(numdvg):
            for vvd in range(numdvd):
                gd0 = 0
                gm0 = 0
                index_ = vvg * (numdvd) + vvd
                if (vvd != (numdvd - 1)):
                    gd0 = (Ids[index_ + 1] - Ids[index_]) / dv
                    gd0 = gd0 * (1 / (np.log(10) * Ids[index_])) #得乘上一个系数 dy/d(vds) = dy/(dI) * (dI/dvds)
                    if (vvg != (numdvg - 1)):
                        gm0 = (Ids[index_ + numdvd] - Ids[index_]) / dv #得乘上一个系数 dy/d(vg) = dy/(dI) * (dI/dvdg)
                        gm0 = gm0 * (1 / (np.log(10) * Ids[index_]))
                    else:
                        gm0 = (Ids[index_] - Ids[index_ - numdvd]) / dv 
                        gm0 = gm0 * (1 / (np.log(10) * Ids[index_]))
                else:
                    gd0 = (Ids[index_] - Ids[index_ - 1]) / dv
                    gd0 = gd0 * (1 / (np.log(10) * Ids[index_]))
                    if (vvg != (numdvg - 1)):
                        gm0 = (Ids[index_ + numdvd] - Ids[index_]) / dv
                        gm0 = gm0 * (1 / (np.log(10) * Ids[index_]))
                    else:
                        gm0 = (Ids[index_] - Ids[index_ - numdvd]) / dv 
                        gm0 = gm0 * (1 / (np.log(10) * Ids[index_ ]))
                
                Gm0.append(gm0)
                Gd0.append(gd0)
        print('Gm0:',Gm0)
        print('Gd0:',Gd0)
            #print('size of Gd/Gm:',len(Gd),len(Gm))
        #print('size of Gd/Gm:',len(Gd),len(Gm))

        ###预处理，具有对称性
        ###考虑到平滑性，不将Vd = 0时 Ids != 0 的不合理情况剔除，在输出端乘上Vds来确保合理性
        u1 = 2 * Vg - Vd
        u2 = [np.log10(np.power(x, 2)) if x!=0 else -10 for x in Vd]
        #print('size of Vg:',len(Vg))
        #print('size of Gd/Gm:',len(Gd),len(Gm))
        #print(Vg[0], Vd[0], Ids[0])
        for k in range(len(Vg)):
            parameter1 = parameter0.copy()
            parameter1 = np.array(parameter1)
            parameter1 = np.log10(parameter1)
            parameter1 = np.append(parameter1, u1[k])
            parameter1 = np.append(parameter1, u2[k])
            parameter1 = np.append(parameter1, Vg[k])
            parameter1 = np.append(parameter1, Vd[k])
            para.append(parameter1)
            target.append(Ids[k])
            Gd.append(Gd0[k])
            Gm.append(Gm0[k])
        #print(len(para))
        #print(len(target))


    return np.array(para), np.array(target), np.array(Gm), np.array(Gd), vdd, vlin, para_name, lowerbound, upperbound