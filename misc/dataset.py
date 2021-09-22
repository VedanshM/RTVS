import torch
torch.manual_seed(0)
from torch.utils import data

class Dataset(data.Dataset):
  def __init__(self, folders):
    self.folders = folders
  
  def __len__(self):
    # For now only 8
    return len(self.folders)
  
  def __getitem__(self, index):
    folder = self.folders[index]
    img_depth = folder + "test.depth.00000.png"
    f12 = read_gen(folder+"flow.flo")
    d1 = plt.imread(img_depth)
    Cx = d1.shape[1]/2
    Cy = d1.shape[0]/2
    kx = 600
    ky = 600
    
    xyz = np.zeros([d1.shape[0],d1.shape[1],3])
    Lsx = np.zeros([d1.shape[0],d1.shape[1],6])
    Lsy = np.zeros([d1.shape[0],d1.shape[1],6])
    
    med = np.median(d1)
    for row in range(xyz.shape[0]):
        for col in range(xyz.shape[1]):
            if(d1[row,col]==0):
                d1[row,col]= med
            xyz[row,col,:] = [(col-Cx)/kx,(row-Cy)/ky,d1[row,col]/10]
            Lsx[row,col,:] =[-1/xyz[row,col,2],0,xyz[row,col,0]/xyz[row,col,2],xyz[row,col,0]*xyz[row,col,1],
                -(1+xyz[row,col,0]**2), xyz[row,col,1]]    
            Lsy[row,col,:] =[0,-1/xyz[row,col,2],xyz[row,col,1]/xyz[row,col,2],(1+xyz[row,col,1]**2),
                -xyz[row,col,0]*xyz[row,col,1], -xyz[row,col,0]]    

    Lps = np.vstack([np.reshape(Lsx,[Lsx.shape[0]*Lsx.shape[1],6]),np.reshape(Lsy,[Lsy.shape[0]*Lsy.shape[1],6])])
    Hps = np.matmul(Lps.T,Lps) + 0.01*np.diag(np.matmul(Lps.T,Lps))
    fps = np.hstack([np.reshape(f12[...,0],[f12.shape[0]*f12.shape[1]]),np.reshape(f12[...,1],[f12.shape[0]*f12.shape[1]])]) 
    vps = - np.matmul(np.linalg.pinv(Lps),fps)

    # Convert to Torch Tensors
    vel = torch.tensor(np.reshape(vps,[1,6]),dtype = torch.float32)
    Lsx = torch.tensor(Lsx, dtype = torch.float32)
    Lsy = torch.tensor(Lsy, dtype = torch.float32)
    f12 = torch.tensor(f12, dtype = torch.float32)
    return vel, Lsx, Lsy, f12, folder