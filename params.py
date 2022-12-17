# Physical para
Temp = 300              # Tempreture [K]
type0 = 1                # NFET: type = 1 PFET: type = -1
esp0 = 8.854187817e-12  # permittivity of vacuum [F/m]
qe = 1.602e-19          # charge of a single electron [Col.]
kb = 8.617e-5           # Boltzmann constant [eV/K]
phit = kb * Temp        # thermal voltage[eV]
esp1 = 3.9              # relative permittivity of SiO2
esp2 = 22               # relative permittivity of HfO2

# Device para
# FinFET structrue para
# height = 20nm, Gate length = 100nm
Width = 8                   #[nm] Fin width
Lg = 100                #[nm] Gate legth
Height = 20                  #[nm] Fin height
Tox1 = 0.4              #[nm] Thickness of SiO2
Cg1 = esp1*esp0/Tox1*1e5 #[F/cm^2]
Tox2 = 3.8              #[nm] Thickness of HfO2
Cg2 = esp2*esp0/Tox2*1e5 #[F/cm^2]
Cg = Cg1*Cg2/(Cg1+Cg2)  #[F/cm^2] Gate capacitance

# GAAFET structrue para
gaa_lg = 12  #[nm] nanowire length
gaa_r = 2    #[nm] nanowire radius
gaa_tox = 0.5   #[nm] Thickness of siO2
gaa_cg = esp1 * esp0 / (gaa_tox) * 1e5 #[F/m^2]

# Empirical para
beta = (type0+1)*1.8/2 - (type0-1)*1.6/2 # Saturation factor. Typ. nFET=1.8, pFET=1.6
alpha = 3.5             # Charge Vt trasistion factor (don't change this mostly between 3.0 and 4.0)

# initial guess of fitting para
delta = 0.15           # DIBL
vxo = 1.2              # VS's velocity [10^7cm/s]
WRs = 100              # parastitic risistance for S and D [Ohms-micron]
mu = 200               # Mobility [cm^2/V·s]
ss = 0.1 * (Temp / 300) # 0.1: SS at T = 27 [V/decade] 
vt0 = 0.4              # threshold voltage. [V]


