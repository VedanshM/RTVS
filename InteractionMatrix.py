import numpy as np
import time

class InteractionMatrix():
    
    def getData(self, f12, d1):
        Cy = d1.shape[1] / 2
        Cx = d1.shape[0] / 2 
        ky = d1.shape[1] / 2
        kx = d1.shape[0] / 2

        xyz = np.zeros([d1.shape[0],d1.shape[1],3])
        Lsx = np.zeros([d1.shape[0],d1.shape[1],6])
        Lsy = np.zeros([d1.shape[0],d1.shape[1],6])
        
        #d1 = d1/255.

        med = np.median(d1)
        for row in range(xyz.shape[0]):
            for col in range(xyz.shape[1]):
                if(d1[row,col]==0):
                    d1[row,col]= med
                xyz[row,col,:] = [(col-Cx)/kx,(row-Cy)/ky,d1[row,col]]
                Lsx[row,col,:] =[-1/xyz[row,col,2],0,xyz[row,col,0]/xyz[row,col,2],xyz[row,col,0]*xyz[row,col,1],
                    -(1+xyz[row,col,0]**2), xyz[row,col,1]]    
                Lsy[row,col,:] =[0,-1/xyz[row,col,2],xyz[row,col,1]/xyz[row,col,2],(1+xyz[row,col,1]**2),
                    -xyz[row,col,0]*xyz[row,col,1], -xyz[row,col,0]]    

        #lamda = 0.01
        #mu = 0.03

        #Lps = np.vstack([np.reshape(Lsx,[Lsx.shape[0]*Lsx.shape[1],6]),np.reshape(Lsy,[Lsy.shape[0]*Lsy.shape[1],6])])
        #H=np.matmul(Lps.T,Lps)
        
        #Hps = np.matmul(Lps.T,Lps) + 0.01*np.diag(np.matmul(Lps.T,Lps))
        #fps = np.hstack([np.reshape(f12[...,0],[f12.shape[0]*f12.shape[1]]),np.reshape(f12[...,1],[f12.shape[0]*f12.shape[1]])]) 
        #vps = - np.matmul(np.linalg.pinv(Lps),fps)
        #vps=-lamda*np.matmul(np.matmul(np.linalg.pinv(H+mu*H.diagonal()),Lps.T),fps)
        return None, Lsx, Lsy
        
    def updateL(self, f12, Lsx, Lsy, d1):
        med = np.median(d1)
        zero_ind = np.where(d1 == 0)[0]        
        d1[zero_ind] = med
        d3 = np.tile(np.expand_dims(d1,-1),[1,1,3])
        
        Lsx[...,:3] /= d3
        Lsy[...,:3] /= d3
        
        #d1 = d1/255.

        
        

        #lamda = 0.01
        #mu = 0.03

        #Lps = np.vstack([np.reshape(Lsx,[Lsx.shape[0]*Lsx.shape[1],6]),np.reshape(Lsy,[Lsy.shape[0]*Lsy.shape[1],6])])
        #H=np.matmul(Lps.T,Lps)
        
        #Hps = np.matmul(Lps.T,Lps) + 0.01*np.diag(np.matmul(Lps.T,Lps))
        #fps = np.hstack([np.reshape(f12[...,0],[f12.shape[0]*f12.shape[1]]),np.reshape(f12[...,1],[f12.shape[0]*f12.shape[1]])]) 
        #vps = - np.matmul(np.linalg.pinv(Lps),fps)
        #vps=-lamda*np.matmul(np.matmul(np.linalg.pinv(H+mu*H.diagonal()),Lps.T),fps)
        return None, Lsx, Lsy



if __name__ == '__main__':
    L = InteractionMatrix()
    f = np.load('flow.npy')
    d = np.load('depth.npy')
    t = time.time()
    # get generic L
    _, Lx0, Ly0 = L.getData(f, np.ones_like(d))
    print("init_time", time.time() - t)
    t = time.time()
    _, Lxz, Lyz = L.updateL(f,Lx0, Ly0, d)
    print("new_time", time.time() - t)
    _, Lx, Ly = L.getData(f, d)
    
    err = np.sum(np.abs(Lxz-Lx))
    print('error = ',err)
    
    #calculate_flow(f, d)
    
    # Interaction matrix computation code
    


