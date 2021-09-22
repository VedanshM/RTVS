import numpy as np
import time 
#https://stackoverflow.com/questions/6254713/create-a-numpy-matrix-with-elements-as-a-function-of-indices



class InteractionMatrix():
    def getData(self, d1, ct, Cy, Cx):
        ky = Cy
        kx = Cx
        t1 = time.time()
        xyz = np.zeros([d1.shape[0],d1.shape[1],3])
        Lsx = np.zeros([d1.shape[0],d1.shape[1],6])
        Lsy = np.zeros([d1.shape[0],d1.shape[1],6])
        
        med = np.median(d1)
        xyz = np.fromfunction(lambda i, j, k : 0.5*(k-1)*(k-2)*(ct*j-float(Cx))/float(kx)- k*(k-2)*(ct*i-float(Cy))/float(ky) + 0.5*k*(k-1)*((d1[i.astype(int), j.astype(int)]==0)*med + d1[i.astype(int), j.astype(int)]), (d1.shape[0], d1.shape[1], 3), dtype=float)
        
        Lsx = np.fromfunction(lambda i, j, k : (k==0).astype(int) * -1/xyz[i.astype(int), j.astype(int),2] + (k==2).astype(int) * xyz[i.astype(int), j.astype(int),0]/xyz[i.astype(int), j.astype(int),2] + (k==3).astype(int) * xyz[i.astype(int),j.astype(int),0]*xyz[i.astype(int),j.astype(int),1] + (k==4).astype(int)*(-(1+xyz[i.astype(int),j.astype(int),0]**2)) +(k==5).astype(int)*xyz[i.astype(int),j.astype(int),1], (d1.shape[0], d1.shape[1], 6), dtype=float)
       
        
        Lsy = np.fromfunction(lambda i, j, k : (k==1).astype(int) * -1/xyz[i.astype(int),j.astype(int),2] + (k==2).astype(int) * xyz[i.astype(int),j.astype(int),1]/xyz[i.astype(int), j.astype(int),2] + (k==3).astype(int) * (1+xyz[i.astype(int),j.astype(int),1]**2) + (k==4).astype(int)*-xyz[i.astype(int),j.astype(int),0]*xyz[i.astype(int),j.astype(int),1] +(k==5).astype(int)* -xyz[i.astype(int),j.astype(int),0], (d1.shape[0], d1.shape[1], 6), dtype=float)
        

        return None, Lsx, Lsy
