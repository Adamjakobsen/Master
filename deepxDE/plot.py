import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib.animation as animation
import matplotlib.tri as tri
from sklearn.preprocessing import MinMaxScaler
from utils import *

from PIL import Image
import io


def plot_2D(pinn, model):

    X, X_boundary, v ,X_train, X_test, v_train, v_test,scaler,scaler_v= pinn.get_data()
    vertices, triangles, vm = get_data()
    # vertices = pinn.vertices
    print("vertices shape", vertices.shape)
    scaler_v = MinMaxScaler()
    # triangles = pinn.triangles
    # vm = pinn.vm
    vm_scaled = scaler_v.fit_transform((vm.reshape(-1, 1)))

    x = vertices[:, 0]
    print("x shape", x.shape)
    y = vertices[:, 1]
    t_ = np.linspace(0, 1, vm.shape[0])
    triang = tri.Triangulation(x, y, triangles)

    for i in [1,10,40]:
        t = np.ones(x.shape)*t_[i]

        X_data = x.reshape(-1, 1)
        Y_data = y.reshape(-1, 1)
        T_data = t.reshape(-1, 1)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_t = MinMaxScaler()
        X_data = scaler_x.fit_transform(X_data)
        Y_data = scaler_y.fit_transform(Y_data)
        # T_data = scaler_t.fit_transform(T_data)
        data = np.hstack((X_data, Y_data, T_data))
        # scaler = MinMaxScaler()
        # data = scaler.fit_transform(data)
        print("t", data[0, -1])
        print("data shape", data.shape)

        v_pred = model.predict(data)[:, 0]
        v_pred = scaler_v.inverse_transform(v_pred.reshape(-1, 1))
        # Plot the triangular mesh with the scalar field
        plt.figure()
        plt.gca().set_aspect('equal')
        # plt.triplot(triang, 'ko-', lw=0.5, alpha=0.5)
        plt.tricontourf(triang, v_pred[:, 0], cmap=plt.cm.jet)
        plt.colorbar(label='Scalar field value')
        plt.title('AP field on Heart Geometry at t={}'.format(t_[i]))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('scalar_field_2D_{}.pdf'.format(i))
    return 0


def generate_2D_animation(pinn, model, frames=int(121*2)):
    X, X_boundary, v = pinn.get_data() #,X_train, X_test, v_train, v_test,scaler,scaler_v = pinn.get_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, v_train, v_test = train_test_split(
        X, v, test_size=0.8)
    vertices = pinn.vertices
    triangles = pinn.triangles
    vm = pinn.vm
    #scaler= MinMaxScaler(feature_range=(0,100))
    #scaler_v = MinMaxScaler()
    #vm_scaled = scaler_v.fit_transform((vm.reshape(-1, 1)))
    #X_train[:,:2] = scaler.fit_transform(X_train[:,:2])
    #v_train = scaler_v.fit_transform(v_train.reshape(-1, 1))
   


    x = vertices[:, 0]
    y = vertices[:, 1]
    scaler = MinMaxScaler()
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    triang = tri.Triangulation(x, y, triangles)
    dt = 5
    
    # Fit the scaler on the initial data
    t_init = np.ones(x.shape) * 0
    X_data_init = x.reshape(-1, 1)
    Y_data_init = y.reshape(-1, 1)
    X_data_init = scaler_x.fit_transform(X_data_init)
    Y_data_init = scaler_y.fit_transform(Y_data_init)
    T_data_init = t_init.reshape(-1, 1)
    data_init = np.hstack((X_data_init, Y_data_init, T_data_init))
    #data_init=scaler.transform(data_init)
    t_ = np.linspace(0, 2, int(121*2))
    X_test = scaler.fit_transform(X_test)
    v_pred_test=model.predict(X_test)[:, 0]
    #v_pred_test = scaler_v.inverse_transform(v_pred_test.reshape(-1, 1))
    #RMSE test
    #RMSE = np.sqrt(np.sum((v_test - v_pred_test)**2)/v_test.shape[0])

    #print("RMSE test", RMSE)
    def update_scalar_field(frame):
        #scaler = MinMaxScaler()
        t = np.ones(x.shape) * t_[frame]
        # print(frame*dt)
        X_data = x.reshape(-1, 1)
        Y_data = y.reshape(-1, 1)
        X_data = scaler_x.transform(X_data)
        Y_data = scaler_y.transform(Y_data)
        T_data = t.reshape(-1, 1)
        data = np.hstack((X_data, Y_data, T_data))
        
        #data = scaler.transform(data)
        # print("t:", data[0, 2])
        v_pred = model.predict(data)[:, 0]
        #v_pred = scaler_v.inverse_transform(v_pred.reshape(-1, 1))
        return v_pred

    def update_plot(frame, plot):
        v_pred = update_scalar_field(frame)
        plot.set_array(v_pred)
        ax.set_title(
            f'Animated AP on heart geometry - Frame: {frame}')
        return plot,

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    v_init = update_scalar_field(1)
    plot = ax.tripcolor(triang, v_init, shading='gouraud', cmap=plt.cm.jet)
    plt.colorbar(plot, label='vm')
    plt.title('Animated AP on heart geometry')
    plt.xlabel('x')
    plt.ylabel('y')

    ani = animation.FuncAnimation(
        fig, update_plot, frames=frames, fargs=(plot,), interval=1, blit=True)

    ani.save('vm_animation_Vnorm_300k_1200ms.mp4',
             writer=animation.FFMpegWriter(fps=10))

    plt.show()

    return 0


def animate_absolute_error(pinn, model, frames=15):
    from sklearn.model_selection import train_test_split
    X, X_boundary, v = pinn.get_data()
    vertices = pinn.vertices
    triangles = pinn.triangles
    vm = pinn.vm_full
    X_train, X_test, v_train, v_test = train_test_split(
        X, v, test_size=0.8)
    print("vm shape", vm.shape)
    scaler_v = MinMaxScaler()
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    vm_scaled = scaler_v.fit_transform((v_train.reshape(-1, 1)))

    x = vertices[:, 0]
    y = vertices[:, 1]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    triang = tri.Triangulation(x, y, triangles)
    print("triang shape", triangles.shape)
    dt = 5
    #scaler = MinMaxScaler()
    # Fit the scaler on the initial data
    t_init = np.ones(x.shape) * 0
    X_data_init = x.reshape(-1, 1)
    Y_data_init = y.reshape(-1, 1)
    #X_data_init = scaler_x.fit_transform(X_data_init)
    #Y_data_init = scaler_y.fit_transform(Y_data_init)
    T_data_init = t_init.reshape(-1, 1)
    data_init = np.hstack((X_data_init, Y_data_init, T_data_init))
    #data_init = scaler.transform(data_init)
    t_ = np.linspace(0, 1, 121)

    def update_error_field(frame):
        #scaler_v = MinMaxScaler()
        t = np.ones(x.shape) * t_[frame]
        # print(frame*dt)
        X_data = x.reshape(-1, 1)
        Y_data = y.reshape(-1, 1)
        #X_data = scaler_x.transform(X_data)
        #Y_data = scaler_y.transform(Y_data)
        T_data = t.reshape(-1, 1)
        data = np.hstack((X_data, Y_data, T_data))
        #data = scaler.transform(data)
        #data = scaler.fit_transform(data)
        # print("t:", data[0, 2])
        v_pred = model.predict(data)[:, 0]

        #v_pred = scaler_v.inverse_transform(v_pred.reshape(-1, 1))
        v_exact = vm[frame, :]#.reshape(-1, 1)
        error = np.abs(v_exact-v_pred)
        #error=v_pred[:,0]
        print("error shape", error.shape,"frame",frame)
        #RMSE
        RMSE = np.sqrt(np.sum((v_pred-v_exact)**2)/v_exact.shape[0])
        print("RMSE", RMSE)
        return error#[:, 0]

    def update_plot(frame, plot):
        print(frame)
        v_error = update_error_field(frame)
        #print("v_error shape", v_error.shape)
        plot.set_array(v_error)
        ax.set_title(
            f'Absolute error of AP on heart geometry - Frame: {frame}')
        return plot,

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    error_init = update_error_field(0)
    plot = ax.tripcolor(triang, error_init, shading='gouraud', cmap=plt.cm.jet)
    plt.colorbar(plot, label='abs error')
    plt.title('Absolute error of AP on heart geometry')
    plt.xlabel('x')
    plt.ylabel('y')

    ani = animation.FuncAnimation(
        fig, update_plot, frames=frames, fargs=(plot,), interval=1, blit=True)

    ani.save('error_field_animation.mp4',
             writer=animation.FFMpegWriter(fps=10))
    print("Animation saved as error_field_animation.mp4")
    #plt.show()

    return 0
