import gc
import deepxde as dde
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import torch as torch
import sys as sys
import scipy.io
import os
#torch.cuda.empty_cache()

# gc.collect()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1024"
# torch.cuda.memory_summary(device=None, abbreviated=False)


class CustomPointCloud(dde.geometry.PointCloud):
    def __init__(self, points, boundary_points, boundary_normals):
        super(CustomPointCloud, self).__init__(
            points, boundary_points, boundary_normals)

    def compute_k_nearest_neighbors(self, x, k=3):
        # Compute the k-nearest neighbors for each boundary point
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='auto').fit(self.boundary_points)
        distances, indices = nbrs.kneighbors(x)
        return indices



    def boundary_normal(self, x):
        """
        k = 3  # number of neighbors
        indices = self.compute_k_nearest_neighbors(x, k)

        normals = np.zeros_like(x)
        for i, idx in enumerate(indices):

            normal = self.boundary_normals[idx[0]]

            normals[i] = normal
        """
        kdtree=KDTree(self.boundary_points)
        distances, idx = kdtree.query(x[0:3], k=1)
        normals=self.boundary_normals[idx.flatten(),:]
        return normals


class PINN():
    def __init__(self):
        self.a = 0.15
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.h = 0.1
        self.D = 0.5#3.333
        self.sigma_a = 1.1151
        self.sigma_b = -0.0634
        self.sigma_c= -0.0634
        self.sigma_d = 1.2102
        self.t_norm = 12.9
        self.touAcm2 = 100/12.9 
        self.vm_norm = 100 #1 voltage unit of U is 100mV
        self.vm_rest = -80 # Resting voltage for U = 0

    def pde2d(self, x, y):
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        # Coupled PDE+ODE Equations
        PDE = dv_dt - self.D*(dv_dxx + dv_dyy) + \
            (self.k*V*(V-self.a)*(V-1) + W*V)*self.touAcm2
        ODE = dw_dt - (self.eps + (self.mu1*W)/(self.mu2+V)) * \
            (-W - self.k*V*(V-self.a-1))/self.t_norm
        return [PDE, ODE]

    def pde2d_(self, x, y):
        V, W = y[:, 0:1], y[:, 1:2]
        
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        # Coupled PDE+ODE Equations
        PDE = dv_dt - self.D*(dv_dxx + dv_dyy) + \
            (self.k*V*(V-self.a)*(V-1) + W*V)#self.touAcm2
        ODE = dw_dt - (self.eps + (self.mu1*W)/(self.mu2+V)) * \
            (-W - self.k*V*(V-self.a-1))/self.t_norm
        return [PDE, ODE]
    
    def pde2d_vm(self, x, y):
        """
        Aliev-Panfilov model with rescaled units

        args: x(tensor): spatial coordinates and time (x,y,t)
                y(tensor): NN output (V,W)
        returns: PDE and ODE residuals
        """
        Vm, W = y[:, 0:1], y[:, 1:2]
        
        V=(Vm -self.vm_rest)/self.vm_norm 
        dv_dt = (dde.grad.jacobian(Vm, x, j=2))
        dv_dxx = (dde.grad.hessian(Vm, x, j=0))
        dv_dyy = (dde.grad.hessian(Vm, x, j=1))
        dw_dt = dde.grad.jacobian(W, x, j=2)
        #dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        #dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        #dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        #dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        # Coupled PDE+ODE Equations
        PDE = dv_dt - self.D*(dv_dxx + dv_dyy) + \
            (self.k*V*(V-self.a)*(V-1) + W*V)*self.touAcm2
        ODE = dw_dt - (self.eps + (self.mu1*W)/(self.mu2+V)) * \
            (-W - self.k*V*(V-self.a-1))/self.t_norm
        return [PDE, ODE]
    
    def pde2d_conductivities(self, x, y):
        """
        Aliev-Panfilov model with rescaled units

        args: x(tensor): spatial coordinates and time (x,y,t)
                y(tensor): solution vector (V,W)
        returns: PDE and ODE residuals
        """
        
        Vm, W = y[:, 0:1], y[:, 1:2]
        V=(Vm -self.vm_rest)/self.vm_norm
        #Get spatial points and convert to numpy
        x_space=x[:,0:2].detach().cpu().numpy()
        
        # Build the KD-tree
        kdtree = KDTree(self.vertices)

        # Query this KD-tree to find the index of the closest point in self.vertices for each point in x_space
        distances, idx = kdtree.query(x_space, k=1)

        # Use these indices to get the corresponding conductivities
        #sigma = self.conductivities[idx.flatten(), :, :]
        #sigma = torch.as_tensor(sigma, dtype=torch.float32)
        #sigma_a=torch.as_tensor(sigma[:,0,0], dtype=torch.float32)
        #sigma_b=torch.as_tensor(sigma[:,0,1],dtype=torch.float32)
        #sigma_c=torch.as_tensor(sigma[:,1,0],dtype=torch.float32)
        #sigma_d=torch.as_tensor(sigma[:,1,1],dtype=torch.float32)
        sigma_a=x[:,3:4]
        sigma_b=x[:,4:5]
        sigma_c=x[:,5:6]
        sigma_d=x[:,6:7]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        #Diffusion term: f=f1+f2=d/dx(a*dv/dx + b*dv/dy) + d/dy(c*dv/dx + d*dv/dy)
        f1= dde.grad.jacobian((sigma_a*dv_dx + sigma_b*dv_dy), x, i=0, j=0) 
        f2= dde.grad.jacobian((sigma_c*dv_dx + sigma_d*dv_dy), x, i=0, j=1)
        f = f1 + f2
        #grad_v = dde.grad.jacobian(V,x)
        #f= dde.grad.jacobian(torch.matmul(sigma,grad_v),x)
        
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        PDE = dv_dt - f + \
            (self.k*V*(V-self.a)*(V-1) + W*V)*self.touAcm2
        
        ODE = dw_dt - (self.eps + (self.mu1*W)/(self.mu2+V)) * \
            (-W - self.k*V*(V-self.b-1))/self.t_norm
        
        
        normals=torch.as_tensor(self.boundary_normal(x), dtype=torch.float32)

        flux = dv_dx*normals[:,0:1] + dv_dy*normals[:,1:2]

        return [PDE, ODE,flux]
    
    
    def get_data(self, slice_xy=2, slice_t=10,t_start=0,extra=False,long=False,isotropic=True):
        """
        Function to load the data,sample the data and scale it

        args: slice_xy(int): sampling rate in x and y direction
                slice_t(int): sampling rate in time direction
        """
        from utils import get_data, get_boundary
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("Loading data...")

        vertices, triangles, vm = get_data(extra=extra,long=long)
        n_timepoints=vm.shape[0]
        dt=5
        t_end=(n_timepoints-1)*dt


        self.vertices = vertices
        self.triangles = triangles

        
        first_ten_vm = vm[0:10, ::slice_xy]
        #last_ten_vm = vm[-10:, ::slice_xy]
        rest_vm = vm[10:121:slice_t, ::slice_xy]
        self.vm= np.concatenate((first_ten_vm, rest_vm), axis=0)
        
        #self.vm = vm[::slice_t, ::slice_xy]
        x = vertices[::slice_xy, 0]
        y = vertices[::slice_xy, 1]
        
        t_all=np.linspace(t_start, t_end, n_timepoints)
        t  = t_all[:121]

        first_ten_t = t[0:10]
        #last_ten_t = t[-10:]
        rest_t = t[10::slice_t]
        t = np.concatenate((first_ten_t, rest_t))
        
        X, T = np.meshgrid(x, t)
        Y, T = np.meshgrid(y, t)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        V = self.vm.reshape(-1, 1)
        vertices_boundary, triangles_boundary = get_boundary()

        #Adding this
        self.boundary_normals = np.load("./data/normals.npy")
        self.boundary_points = vertices_boundary

        self.vertices_boundary = vertices_boundary
        self.triangles_boundary = triangles_boundary

        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]
        X_boundary, T_boundary = np.meshgrid(x_boundary, t)
        Y_boundary, T_boundary = np.meshgrid(y_boundary, t)
        X_boundary = X_boundary.reshape(-1, 1)
        T_boundary = T_boundary.reshape(-1, 1)
        Y_boundary = Y_boundary.reshape(-1, 1)
        ones=np.ones_like(X_boundary)

        self.conductivities=np.load('./data/conductivities.npy')
        sigma_a=self.conductivities[::slice_xy,0,0]
        sigma_b=self.conductivities[::slice_xy,0,1]
        sigma_c=self.conductivities[::slice_xy,1,0]
        sigma_d=self.conductivities[::slice_xy,1,1]
        sigma_a,T_ = np.meshgrid(sigma_a, t)
        sigma_b,T_ = np.meshgrid(sigma_b, t)
        sigma_c,T_ = np.meshgrid(sigma_c, t)
        sigma_d,T_ = np.meshgrid(sigma_d, t)
        sigma_a = sigma_a.reshape(-1, 1)
        sigma_b = sigma_b.reshape(-1, 1)
        sigma_c = sigma_c.reshape(-1, 1)
        sigma_d = sigma_d.reshape(-1, 1)
        if not isotropic:
            return np.hstack((X, Y, T,sigma_a,sigma_b,sigma_c,sigma_d)), np.hstack((X_boundary, Y_boundary, T_boundary,ones,ones,ones,ones)), V 

        else:
            return np.hstack((X, Y, T)), np.hstack((X_boundary, Y_boundary, T_boundary)), V
        
    def boundary_normal(self, x):
        
        x_cpu = x[:,0:2].detach().cpu().numpy()

        kdtree = KDTree(self.boundary_points)
        distances, idx = kdtree.query(x_cpu, k=1)
        normals = self.boundary_normals[idx.flatten(), :]
        return normals

    def BC(self, geomtime):
        """No flux von Neumann boundary condition"""
        #bc = dde.NeumannBC(geomtime, lambda x:  np.zeros(
        #    (len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        #bc = dde.icbc.OperatorBC()
        return bc
    
    def BC_custom(self, x,y):
        """ No flux von neumann using pointsetbc"""
        normals=self.boundary_normal(x)
        dV_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dV_dy = dde.grad.jacobian(y, x, i=0, j=1)
        flux = dV_dx*normals[:,0:1] + dV_dy*normals[:,1:2]
        return flux
    

    def IC(self, observe_train, v_train,observe_train_unscaled):
        observe_train=observe_train#[0:3,:]
        observe_train_unscaled=observe_train_unscaled#[0:3,:]
        T_ic = observe_train_unscaled[:, 2:3].reshape(-1, 1)

        idx_init = np.where(np.isclose(T_ic, 5, rtol=6))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]

        return dde.PointSetBC(observe_init, v_init, component=0)

    def geotime(self):

        self.boundary_normals = np.load("./data/normals.npy")
        # remove points from vertices that are on the boundary
        vertices_expanded = self.vertices[:, np.newaxis]
        boundary_vertices_expanded = self.vertices_boundary[np.newaxis, :]

        is_vertex_on_boundary = np.any(
            np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1)
        self.unique_vertices = self.vertices[~is_vertex_on_boundary]

        geom = CustomPointCloud(
            self.unique_vertices, self.vertices_boundary, self.boundary_normals)
        timedomain = dde.geometry.TimeDomain(0, 600)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        return geomtime