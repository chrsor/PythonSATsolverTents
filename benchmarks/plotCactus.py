import numpy as np
import matplotlib.pyplot as plt


# data format : [[filename, result, spent_time, solution],...]
vsdis = np.load('VSIDS-100-None.npy', allow_pickle=True)
print(vsdis[vsdis[:, 3] == None])
vsdis = vsdis[vsdis[:, 3] != None]
vsdis = vsdis[vsdis[:, 0].argsort()]
vsdis_time = vsdis[:, 3].astype(np.float)
vsdis_time = np.cumsum(vsdis_time)

vmtf = np.load('VMTF-100-None.npy', allow_pickle=True)
print('------')
print(vmtf[vmtf[:, 3] == None])
vmtf = vmtf[vmtf[:, 3] != None]
vmtf = vmtf[vmtf[:, 0].argsort()]
vmtf_time = vmtf[:, 3].astype(np.float)
vmtf_time = np.cumsum(vmtf_time)

vsdis_n = np.load('VSIDS-None-None.npy', allow_pickle=True)
print(vsdis_n[vsdis_n[:, 3] == None])
vsdis_n = vsdis_n[vsdis_n[:, 3] != None]
vsdis_n = vsdis_n[vsdis_n[:, 0].argsort()]
vsdis_n_time = vsdis_n[:, 3].astype(np.float)
vsdis_n_time = np.cumsum(vsdis_n_time)


vmtf_n = np.load('VMTF-None-None.npy', allow_pickle=True)
print('------')
print(vmtf_n[vmtf_n[:, 3] == None])
vmtf_n = vmtf_n[vmtf_n[:, 3] != None]
vmtf_n = vmtf_n[vmtf_n[:, 0].argsort()]
vmtf_n_time = vmtf_n[:, 3].astype(np.float)
vmtf_n_time = np.cumsum(vmtf_n_time)

vsdis_n_p = np.load('VSIDS-100-True.npy', allow_pickle=True)
print(vsdis_n_p[vsdis_n_p[:, 3] == None])
vsdis_n_p = vsdis_n_p[vsdis_n_p[:, 3] != None]
vsdis_n_p = vsdis_n_p[vsdis_n_p[:, 0].argsort()]
vsdis_n_p_time = vsdis_n_p[:, 3].astype(np.float)
vsdis_n_p_time = np.cumsum(vsdis_n_p_time)

plt.figure(figsize=(20, 10))
plt.ylabel('CPU time (s)')
plt.xlabel('Solved instances')
plt.ylim(-0.1, 400)
plt.xlim(-0.1, 172)

plt.plot(vsdis_n_time)
plt.scatter(range(len(vsdis_n_time)), vsdis_n_time, s=20, marker='x', label='VSDIS')

plt.plot(vmtf_n_time)
plt.scatter(range(len(vmtf_n_time)), vmtf_n_time, s=20, marker='x', label='VMTF')

plt.plot(vsdis_time)
plt.scatter(range(len(vsdis_time)), vsdis_time, s=20, marker='x', label='VSDIS+Restart')

plt.plot(vmtf_time)
plt.scatter(range(len(vmtf_time)), vmtf_time, s=20, marker='x', label='VMTF+Restart')

plt.plot(vsdis_n_p_time)
plt.scatter(range(len(vsdis_n_p_time)), vsdis_n_p_time, s=20, marker='x', label='VMTF+Restart+Preprocessing')

plt.legend()

plt.savefig("cactus_plot_all.png")
