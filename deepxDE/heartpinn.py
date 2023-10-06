import gc
import deepxde as dde
import numpy as np

from sklearn.model_selection import train_test_split
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
        k = 3  # number of neighbors
        indices = self.compute_k_nearest_neighbors(x, k)

        normals = np.zeros_like(x)
        for i, idx in enumerate(indices):

            normal = self.boundary_normals[idx[0]]

            normals[i] = normal

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
        self.D = 3.333
        self.t_norm = 12.9
        self.touAcm2 = 100/12.9 #tried only this one
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
        Vm, W = y[:, 0:1], y[:, 1:2]
        V=(Vm -self.vm_rest)/self.vm_norm 
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
    
    def get_data(self):
        from utils import get_data, get_boundary
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        vertices, triangles, vm = get_data()

        self.vertices = vertices
        self.triangles = triangles

        print("self.triangles shape:", self.triangles.shape)
        self.vm = vm[::10, ::2]
        x = vertices[::2, 0]
        y = vertices[::2, 1]
        t = np.linspace(0, 600, 121)[::10]

        X, T = np.meshgrid(x, t)
        Y, T = np.meshgrid(y, t)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        V = self.vm.reshape(-1, 1)
        vertices_boundary, triangles_boundary = get_boundary()

        self.vertices_boundary = vertices_boundary
        self.triangles_boundary = triangles_boundary

        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]
        X_boundary, T_boundary = np.meshgrid(x_boundary, t)
        Y_boundary, T_boundary = np.meshgrid(y_boundary, t)
        X_boundary = X_boundary.reshape(-1, 1)
        T_boundary = T_boundary.reshape(-1, 1)
        Y_boundary = Y_boundary.reshape(-1, 1)

        return np.hstack((X, Y, T)), np.hstack((X_boundary, Y_boundary, T_boundary)), V

    def BC(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros(
            (len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        return bc

    def IC(self, observe_train, v_train):

        T_ic = observe_train[:, -1].reshape(-1, 1)

        idx_init = np.where(np.isclose(T_ic, 5, rtol=1))[0]
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