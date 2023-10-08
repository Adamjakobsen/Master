import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib.animation as animation
import matplotlib.tri as tri
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import deepxde as dde
from heartpinn import CustomPointCloud, PINN
from utils import *


def plot_animation(pinn,model,frames,experiment_path,filename):
    X,X_boundary,v =pinn.get_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    vertices = pinn.vertices
    triangles = pinn.triangles

    x=vertices[:,0]
    y=vertices[:,1]
    triang = tri.Triangulation(x, y, triangles)
    t=np.linspace(0,2,frames)

    t_init = np.ones(x.shape)*t[1]
    X_data_init = x.reshape(-1, 1)
    Y_data_init = y.reshape(-1, 1)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_data_init = scaler_x.fit_transform(X_data_init)
    Y_data_init = scaler_y.fit_transform(Y_data_init)
    T_data_init = t_init.reshape(-1, 1)
    data_init = np.hstack((X_data_init, Y_data_init, T_data_init))
    v_init = model.predict(data_init)[:,0]
    

    def update_scalar_field(frame):
        t_ = np.ones(x.shape)*t[frame]
        #print(frame*dt)
        X_data = x.reshape(-1, 1)
        Y_data = y.reshape(-1, 1)
        X_data = scaler_x.transform(X_data)
        Y_data = scaler_y.transform(Y_data)
        T_data = t_.reshape(-1, 1)
        data = np.hstack((X_data, Y_data, T_data))
        
        v_pred = model.predict(data)[:,0]
        
        return v_pred
    
    def update_plot(frame,plot):
        v_pred = update_scalar_field(frame)
        plot.set_array(v_pred)
        ax.set_title(f"t = {t[frame]:.2f} ms")
        return plot,


    micro_to_milli = 1e-3

    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    plot = ax.tripcolor(triang,v_init,shading='gouraud', cmap=plt.cm.jet)
    plt.colorbar(plot,label = "vm")
    plt.xlabel("x")
    plt.ylabel("y")

    ani = animation.FuncAnimation(
        fig, update_plot, frames=frames, fargs=(plot,), interval=1, blit=True)
    path = os.path.join(experiment_path,filename)
    ani.save(path,writer=animation.FFMpegWriter(fps=10))

    return 0


def get_model(config):
    # Load the configuration
    n_neurons = config["TRAINING"]["n_neurons"]
    n_layers = config["TRAINING"]["n_layers"]
    activation = config["TRAINING"]["activation"]
    initializer = config["TRAINING"]["initializer"]
    test_size = config["DATA"]["test_size"]
    num_domain = config["DATA"]["num_domain"]
    num_boundary = config["DATA"]["num_boundary"]
    resampling_period = config["DATA"]["resampling_period"]
    #batch_size = config["DATA"]["batch_size"]

    pinn = PINN()
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    # Load the data and scale
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=test_size, random_state=42)
    data_list = [X_train, X_test, v_train, v_test]
    # Initialize the model
    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    ic = pinn.IC(X_train, v_train)
    bc = pinn.BC(geomtime)
    input_data = [bc, ic, observe_v]
    PDE= pinn.pde2d_vm
    data = dde.data.TimePDE(geomtime,
                            PDE,
                            input_data,
                            num_domain=num_domain,
                            num_boundary=num_boundary)
    
    net = dde.maps.FNN([3] + n_layers * [n_neurons] +
                       [2], activation, initializer)
    

    model = dde.Model(data, net)
    return model,net,pinn

def main():
    
    #2023-10-06 12:43_128x5_tanh_Glorot normal_numdom2000_rs10000
    #2023-10-06 14:20_128x5_tanh_Glorot normal_numdom2000_rs10000
    #/itf-fi-ml/home/adamjak/heart/MSc/deepxDE/experiments/2023-10-06 11:51_128x6_tanh_Glorot normal_numdom2000_rs10000
    #/itf-fi-ml/home/adamjak/heart/MSc/deepxDE/old_stuff/models/heart_model_64x5_tanh_Glorot normal_mini_t_norm-160000.pt
    experiment_path = "./experiments/2023-10-06 11:51_128x6_tanh_Glorot normal_numdom2000_rs10000"
    config_path = os.path.join(experiment_path,"config.json")
    #/itf-fi-ml/home/adamjak/heart/MSc/deepxDE/old_stuff/models/heart_model_64x5_tanh_Glorot normal-200000.pt
    #experiment_path = "./old_stuff/models/heart_model_64x5_tanh_Glorot normal-200000.pt"
    #pt_file=torch.load(experiment_path)
    #print(pt_file)
    #config_path= "config.json"
    config = load_config(config_path)
    model,net,pinn = get_model(config)
    model.compile("adam", lr=config["TRAINING"]["lr_phase1"])
    model.restore(os.path.join(experiment_path,"model-165000.pt"))
    #model.restore(experiment_path)
    
    plot_animation(pinn,model,frames=241,experiment_path=experiment_path,filename="vm_animation.mp4")
    #plot_animation(pinn,model,frames=121,experiment_path="./old_stuff",filename="vm_animation.mp4")
    return 0

if __name__ == "__main__":
    main()










    
