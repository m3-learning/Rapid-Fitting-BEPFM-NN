import numpy as np
import torch 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec

from belearn.functions.sho import SHO_nn
from belearn.functions.hysteresis import hysteresis_nn

from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model
from autophyslearn.postprocessing.complex import ComplexPostProcessor

from m3util.ml.rand import set_seeds
from m3util.viz.text import set_sci_notation_label, labelfigs
from m3util.viz.layout import FigDimConverter
from m3util.ml.optimizers.TrustRegion import TRCG



def clims():
    clims_ = [
            (0, 1.4e-4),  # amplitude
            (1.31e6, 1.33e6),  # resonance frequency
            (-240, -160),  # quality factor
            (-np.pi, np.pi),  # phase
                ],  # phase limits
    return clims_

def instantiate_SHO_model(visualizer,
                        noise = 0,
                        Train = False, 
                        model_basename = "SHO_Fitter_original_data_noise_0", 
                        datafed_path = '2024_SHO_Fitting/Noisy_NN',
                        script_path = './Paper_Figures.ipynb',
                        seed=42, 
                        device = 'cuda:0'
                        ):
    
    #visualizer.noise = noise
    print("instantiating model for noise level", visualizer.noise)
    visualizer.get_dataset(noise = visualizer.noise)
    visualizer.SHO_preprocessing() 
        
    set_seeds(seed)
    postprocessor = ComplexPostProcessor(visualizer,device=device)

    model_ = Multiscale1DFitter(SHO_nn, # function 
                        visualizer.frequency_bin, # x data
                        2, # input channels
                        4, # output channels
                        visualizer.SHO_scaler, 
                        postprocessor,
                        device = device)
    
    #TODO: I am able to load the weights just fine instead of training with training=True
    # so maybe I don't need to be able to set training=False?  
    
    model = Model(model_, visualizer, training=False,
            model_basename=model_basename,
            datafed_path=datafed_path,
            script_path = script_path,
            device = device)
    
    X_train, X_test, y_train, y_test = visualizer.test_train_split_(shuffle=True)

    Train = False

    if Train: 
        model.fit(
        X_train,
        500,
        optimizer="Adam",
        epochs = 5,
        )

    else:
        if noise == 0:
            model.load(
                "./Trained Models/SHO Fitter/2024-09-23_14-36-21_nn_benchmarks_noise/SHO_Fitter_model_optimizer_Adam_epoch_4_train_loss_0.040321211942850994.pth",
                device=device
            )
        elif noise == 2:
            model.load(
                "./Trained Models/SHO Fitter/SHO_Fitter_original_data_noise_2_model_optimizer_Adam_epoch_4_train_loss_0.7284429110349501.pth",
                device=device
            )
        elif noise == 4:
            model.load(
                "./Trained Models/SHO Fitter/SHO_Fitter_original_data_noise_4_model_optimizer_Adam_epoch_4_train_loss_0.9066618817979125.pth",
                device = device
            )
        elif noise == 7:
            model.load(
                "./Trained Models/SHO Fitter/SHO_Fitter_original_data_noise_7_model_optimizer_Adam_epoch_4_train_loss_0.9597052748506906.pth",
                device = device
            )
        else: 
            raise ValueError(f"Noise level {noise} has not been trained yet. \n Please train the model before loading it.")
    return model


def instantiate_hysteresis_model(visualizer,
                                noise = 0,
                                Train = False, 
                                model_basename = "hysteresis_Fitter_original_data_noise_0", 
                                datafed_path = '2024_SHO_Fitting/Noisy_NN',
                                script_path = './Paper_Figures.ipynb',
                                seed=42, 
                                device = 'cuda:0'):
    set_seeds(seed)
    

    data, voltage = visualizer.get_hysteresis(scaled=True, loop_interpolated = True)
    # V = np.swapaxes(np.atleast_2d(dataset.get_voltage), 0, 1).astype(np.float64)

    data = torch.atleast_3d(torch.tensor(data.reshape(-1, 96)))

    model_ = Multiscale1DFitter(
                #BE_viz.loop_fitting_function_torch, # function 
                    hysteresis_nn,  # function

                                voltage[:,0].squeeze(), # x data
    #                             V.squeeze(),
                                1, # input channels
                                9, # output parameters
                                visualizer.loop_param_scaler,
                                loops_scaler=visualizer.hysteresis_scaler,
                                device=device
                                )

    # instantiate the model
    model = Model(model_, visualizer, training=Train, model_basename=model_basename,
                    datafed_path=datafed_path,
                    script_path=script_path,
                    device=device)





    X_train, X_test = train_test_split(data.reshape(-1,96), test_size=0.2, random_state=42, shuffle=True)

    X_train = np.atleast_3d(X_train)

    optimizer = {
        "name": "TRCG", 
        "optimizer": TRCG,
        "closure_size": 1,
        "cgopttol": 1e-3,
        "c0tr": 0.2,
        "c1tr": 0.25,
        "c2tr": 0.75,
        "t1tr": 0.75,
        "t2tr": 2.0,
        "radius_max": 5.0,  
        "radius_initial": 1.0,
        "radius" : 1.0,
        "device": device,
        "ADAM_epochs": 100}


    if Train:
        # fits the model
        model.fit(
            X_train,
            1024,
            optimizer=optimizer,
            epochs = 5,
        )
    else:
        model.load(
        # "./Trained Models/SHO Fitter/SHO_Fitter_original_data_model_epoch_5_train_loss_0.0449272525189978.pth"
        "./Trained Models/SHO Fitter/SHO_Fitter_original_data_model_optimizer_Trust Region CG_epoch_499_train_loss_0.005630633379850123.pth"
        )
    
    return model, data
    
    

def instantiate_SHO_model_params(visualizer,model):
    
    X_data, Y_data = visualizer.get_nn_data()
    NN_recon, NN_params_scaled, NN_params = model.predict(X_data)

    #return X_data, Y_data, pred_data, scaled_param, NN_params
    return X_data, NN_params


def instantiate_hysteresis_model_params(visualizer,model,data):
    
    NN_recon, NN_params_scaled, NN_params = model.predict(
    data,
    1024,
    translate_params=False,
    is_SHO=False
)
    return NN_params


def copy_axis_to(source_ax, target_ax):
    """
    Copies the plot content from source_ax into target_ax, including lines, markers,
    bar plots, annotations, and text.

    Args:
        source_ax (matplotlib.axes.Axes): The axis to copy from.
        target_ax (matplotlib.axes.Axes): The axis to copy into.
    """
    # Copy lines and markers
    for line in source_ax.get_lines():
        target_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            color=line.get_color(),
            label=line.get_label(),
            markersize=line.get_markersize()
        )

    # Copy bar containers
    for container in source_ax.containers:
        for patch in container:
            target_ax.add_patch(patch)

    # Copy text and annotations
    for txt in source_ax.texts:
        if isinstance(txt, plt.Annotation):
            # Copy annotation with arrows (from annotate)
            target_ax.annotate(
                text=txt.get_text(), 
                xy=txt.xy,
                xytext=txt.get_position(),
                arrowprops=txt.arrowprops if hasattr(txt, 'arrowprops') else None
            )
        else:
            # Copy plain text (from text)
            target_ax.text(
                x=txt.get_position()[0],
                y=txt.get_position()[1],
                s=txt.get_text(),
                fontsize=txt.get_fontsize(),
                color=txt.get_color(),
                ha=txt.get_ha(),
                va=txt.get_va(),
                rotation=txt.get_rotation()
            )


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    if abs(b) >2: 
        #return r'${} \times 10^{{{}}}$'.format(a, b)
        return rf'$\hspace{{{-1.1}}} {a} \hspace{{{-0.4}}} \times \hspace{{{-0.4}}} 10^{{{b}}}$'

    else: 
        return float(a)*10**b
    
def fmt_resonance(x, pos): 
    #need to display more digits to differentiate the resonance values
    # for the resonance, b=6 so we don't need to worry about abs(b)<2
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return rf'$\hspace{{{-1.1}}} {a} \hspace{{{-0.4}}} \times \hspace{{{-0.4}}} 10^{{{b}}}$'
                            

def y_formatter(y, pos):
    return f"{y * 1e3:.1f}"  # Multiply by 1e3 to show scaled values

def copy_axes_properties(row,col,source_ax, target_ax, secondary_ax, ax_lims):
    """Copy properties and data from source_ax to target_ax."""
    # Copy basic properties
    target_ax.set_xlim([1.2,1.4])
    target_ax.set_xticks([1.2,1.3,1.4])
    target_ax.get_xticklabels()[0].set_horizontalalignment('left')
    target_ax.get_xticklabels()[-1].set_horizontalalignment('right')

    if row == 2: # if i in [4,5]:
        target_ax.set_xlabel('Frequency (MHz)',fontsize=20)
    else:
        target_ax.set_xlabel("")
        #target_ax.set_xticks([])
        target_ax.xaxis.set_ticklabels([])

    if col == 0:    
        target_ax.set_ylabel(source_ax.get_ylabel(),fontsize=20)
    else:
        target_ax.set_ylabel("")
        target_ax.yaxis.set_ticklabels([])

        
    if row == 0:
        target_ax.set_ylim([-0.5e-3,8.0e-3])
    elif row == 1: 
        target_ax.set_ylim([-0.1e-2,2.1e-2])
    elif row == 2: 
        target_ax.set_ylim([-0.1e-2,2.1e-2])    
    
    target_ax.set_title(source_ax.get_title())

    # Copy lines from the primary axis
    for line in source_ax.get_lines():
        label = line.get_label() if line.get_label() != '_nolegend_' else None
        # Copying line properties like color, linestyle, marker, etc.
        target_ax.plot(line.get_xdata()/1e6, line.get_ydata(), color=line.get_color(),
                    linestyle=line.get_linestyle(), marker=line.get_marker(), label=label)
        
    # Handle twin axes if present
    #if secondary_ax:
    ax_twin = target_ax.twinx()
    ax_twin.set_ylim(secondary_ax.get_ylim())

    if col == 0: #if i % 2 == 0: 
        ax_twin.set_ylabel("")
        ax_twin.yaxis.set_ticklabels([])
        
        set_sci_notation_label(
                target_ax, corner="top left", axis="y", stroke_color="w", linewidth=0.5,
                textsize = 20, offset_points = (0,50)
            )
    else:
        ax_twin.set_ylabel(secondary_ax.get_ylabel(),fontsize = 20)
        ax_twin.set_yticks([-3,-2,-1,0,1,2,3])


    for line in secondary_ax.get_lines():
        label = line.get_label() if line.get_label() != '_nolegend_' else None
        # Copying line properties for the twin axis
        ax_twin.plot(line.get_xdata()/1e6, line.get_ydata(), color=line.get_color(),
                        linestyle=line.get_linestyle(), marker=line.get_marker(), label=label)
    
    
    target_ax.tick_params(axis='x',labelsize=18)
    target_ax.tick_params(axis='y',labelsize=18)
    if row == 0 and col == 0: 
        target_ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

    ax_twin.tick_params(axis='x',labelsize=18)

    ax_twin.tick_params(axis='y',labelsize=18)
    
    plt.tight_layout()

                                
def plot_figure_3(visualizer, filename = None):
    """
    Plots the figure 3 of the paper.
    """
    
    

    
    # true state for the violin plot and BWM fit comparison
    true_state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": True,
        "scaled": True,
        "output_shape": "index",
        "measurement_state": "all",
        "LSQF_phase_shift": np.pi/2,
        "NN_phase_shift": np.pi/2,
        "noise": 0
    }
    
    LSQF_state = {'resampled': True,
            'raw_format': 'complex',
            'fitter': 'LSQF',
            'scaled': False,
            'output_shape': 'index',
            'measurement_state': 'all',
            'resampled_bins': 165,
            'LSQF_phase_shift': 1.5707963267948966,
            'NN_phase_shift': 1.5707963267948966,
            'noise': 0}

    LSQF_params = visualizer.SHO_fit_results(state = LSQF_state)
    
    fig = plt.figure(figsize=(24, 24))


    # Define the GridSpec layout
    gs = GridSpec(60, 40, figure=fig)

    order = [['SHO_fit_comp'],
            ['violin'],
            ['voltage_curve'],
            ['switching_maps']
            ]



    subplot_specs = [(0, 30, 0, 20 ), # top left: SHO fit comparisons 
                    (0, 15, 20, 40), # g: violin plot
                    (17, 27, 20, 40), # h: voltage curve
                    (30, 80, 0, 60), #bottom: switching maps
                    ]

    for i, (r_start, r_end, c_start, c_end) in enumerate(subplot_specs):
        ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
        idx = order[i]
        
        if idx[0] == 'SHO_fit_comp':
            ax.axis("off")
            
            model = instantiate_SHO_model(visualizer, Train = False)
            # sets the state of the output data
            out_state = {"scaled": True, "raw_format": "magnitude spectrum"}
            
            LSQF_data = visualizer.get_best_median_worst(
            true_state,
            prediction={"fitter": "LSQF"},
            #model = model,
            out_state=out_state,
            SHO_results=True,
            n=1,
            )
            NN_data = visualizer.get_best_median_worst(
                true_state, prediction=model, out_state=out_state, SHO_results=True, n=1
            )

            data = (LSQF_data, NN_data)
            model_names = ["LSQF", "NN"]
    
            
            BMW_comp_fig,list_ax_,list_ax1 = visualizer.SHO_Fit_comparison(
                data=data,
                names=model_names,
                model_comparison=[model, {"fitter": "LSQF"}],
                out_state=out_state,
                filename = None,
                SHO_results=True,
                n=1,
            )
            
            # the order of the plots in the paper seem to be different 
            # from the order of the plots in the code.
            # To make them match, put the plots in the following order: 
            axes_index = [0,3,1,4,2,5]
                    
            for row in range(3):
                for col in range(2):
                    inset_ax = ax.inset_axes([(col/2)-col*0.18,1-(row+1)/3.25,1/3.25,1/3.25])
                    if col == 0: 
                        copy_axes_properties(row,col,list_ax_[axes_index[2*row+col]], inset_ax, list_ax1[axes_index[2*row+col]],list_ax_[axes_index[2*row+col+1]])
                    else:
                        copy_axes_properties(row,col,list_ax_[axes_index[2*row+col]], inset_ax, list_ax1[axes_index[2*row+col]],list_ax_[axes_index[2*row+col-1]])
                
                    if row == 0:
                        labelfigs(inset_ax,
                                string_add="Best",
                                loc ='ct',
                                label_size=20,
                                inset_fraction=(0.075,0.5),
                                style = 'b',
                                horizontalalignment = "center"
                                )
                        if col == 1:
                            # get legend handles and their corresponding labels
                            handles1, labels1 = inset_ax.get_legend_handles_labels()
                            handles2, labels2 = list_ax1[axes_index[2*row+col]].get_legend_handles_labels()
                            
                            inset_ax.legend(handles1 + handles2,labels1+labels2, loc=(1.25,0.45),fontsize = 14)

                    elif row == 1:
                        labelfigs(inset_ax,
                                string_add="Median",
                                loc ='ct',
                                label_size=20,
                                inset_fraction=(0.075,0.5),
                                style = 'b',
                                horizontalalignment = "center"
                                )
                        
                    elif row == 2:
                        labelfigs(inset_ax,
                                string_add="Worst",
                                loc ='ct',
                                label_size=20,
                                inset_fraction=(0.075,0.5),
                                style = 'b',
                                horizontalalignment="center"
                                )
                        ax.set_xticks([1.2,1.3,1.4])
                        
                    
                    labelfigs(inset_ax,
                        number=2*row+col,
                        loc ='tr',
                        label_size=20,
                        inset_fraction=(0.075,0.075),
                        style = 'b'
                        )
            plt.close(BMW_comp_fig)
        
        elif idx[0] == 'violin':
            X_data, Y_data = visualizer.get_nn_data()
            pred_data, scaled_param, NN_params = model.predict(X_data)
            
            visualizer.violin_plot_comparison_SHO(
                true_state,
                model,
                X_data,
                filename=None,
                label="NN",
                ax=ax,
                figlabel='g',
                label_size=20,
                loc = 'tr',
                inset_fraction=(0.075,0.075)
            )
            
            ax.set_ylabel("Scaled SHO Results",fontsize=20)
            ax.set_xlabel("")
            
            ax.tick_params(axis='x',labelsize=20)
            ax.tick_params(axis='y',labelsize=20)
            ax.set_yticks(np.linspace(-6,6,7))


            # Get the legend associated with the plot
            legend = ax.get_legend()
            legend.set_title("")
            plt.setp(legend.get_texts(), fontsize=20) # Set the label size
            
        elif idx[0] == 'voltage_curve':
            voltage_and_switching_maps_fig = visualizer.SHO_switching_maps(SHO_ = [LSQF_params,NN_params],
                                            labels = ["LSQF", "NN"], 
                                            filename=None,
                                            label_marker_starting_index=8,
                                            label_marker_size=16,
                                            label_letter_text_size=18,
                                            colorbars=False,
                                            )
            copy_axis_to(voltage_and_switching_maps_fig.axes[0], ax)  
            
            ax.set_ylabel("Voltage (V)",fontsize=20)
            ax.set_xlabel("Step",fontsize=20)
            ax.set_xticks(np.linspace(0,100,11))
            ax.set_xlim([-4,100])
            ax.set_ylim([-20,20])
            ax.set_yticks([-15,0,15])
            ax.tick_params(axis='x',labelsize=20)
            ax.tick_params(axis='y',labelsize=20)
            
            labelfigs(ax,
                string_add='h',
                loc ='tr',
                label_size=20, #22
                inset_fraction=(0.12,0.04),
                style = 'b'
                )

            plt.tight_layout()

        else: # idx[0] == 'switching_maps':
            ax2 = voltage_and_switching_maps_fig.axes[1:]
            
            label_marker_symbols = ["\u25CF", "\u25BC", "\u25B2", "\u25BA", "\u25C0", "\u25A0","\u271A", "\u25C6","\u2605"]
            label_marker_symbols_counter = 0
            
            
            labels = ['i','j','k','l','m','n','o','p','q']
            label_counter = 0
            names = ['Amplitude', "Resonance","Quality Factor","Phase"]
            
            # defines a scalar to convert inches to relative coordinates
            fig_scalar = FigDimConverter((1/6, 1/6))
            
            for row in range(6):
                for col in range(12):
                    inset_ax = ax.inset_axes([-0.06+(col/11.8)+np.floor(col/4)/256,1-(row+1)/6.1-row/192-np.floor(row/2)/96,1/6.1,1/6.1])

                    inset_ax.imshow(ax2[(12*row)+col].get_images()[0].get_array().data,clim = clims()[0][int(np.floor(col % 4))])
                    if col == 0:
                        if row % 2 == 0: 
                            inset_ax.text(35,10,ax2[0].get_ylabel(),color = "white",size=20,ha = "center", va = "center")
                        
                        else:
                            inset_ax.text(35,10,ax2[12].get_ylabel(),color = "white", size=20,ha = "center", va = "center")
                    
                    if row % 2 == 0 and col % 4 == 0: 
                            inset_ax.text(10,10, label_marker_symbols[label_marker_symbols_counter], color = "white", size = 24, ha = "center", va = "center")
                            label_marker_symbols_counter+=1
                    elif row % 2 == 1 and (col + 1) % 4 == 0:
                        inset_ax.text(50,50, labels[label_counter], color = "white", weight = 'bold', size = 20,ha = "center", va = "center")
                        label_counter+=1 
                        
                    if row == 5:    
                        bar_ax = []
                        pos_inch = [-3.2e-3 + (col/70.5)+np.floor(col/4)/1800, -0.008, 1/73.5, 1/500  ] #fills axes
                        
                        bar_ax.append(ax.inset_axes(fig_scalar.to_relative(pos_inch)))
                                        
                        if int(np.floor(col % 4)) == 0:
                                                
                            cbar = plt.colorbar(inset_ax.images[0],location = 'bottom', cax = bar_ax[0], 
                                                format = FuncFormatter(fmt), 
                                                ticks = np.linspace(np.min(clims()[0][int(np.floor(col % 4))]),
                                                                    np.max(clims()[0][int(np.floor(col % 4))]),2), #5
                                                
                                                )
                            cbar.ax.get_xticklabels()[0].set_horizontalalignment('left')
                            cbar.ax.get_xticklabels()[1].set_horizontalalignment('right')

                        elif int(np.floor(col % 4)) == 1:
                            cbar = plt.colorbar(inset_ax.images[0],location = 'bottom', cax = bar_ax[0], 
                                                format = FuncFormatter(fmt_resonance), #fmt, 
                                                ticks = np.linspace(np.min(clims()[0][int(np.floor(col % 4))]),
                                                                    np.max(clims()[0][int(np.floor(col % 4))]),2) #5
                                                )
                        elif int(np.floor(col % 4)) == 2:
                            cbar = plt.colorbar(inset_ax.images[0],location = 'bottom', cax = bar_ax[0], 
                                                format = FuncFormatter(fmt), 
                                                ticks = np.linspace(np.min(clims()[0][int(np.floor(col % 4))]),
                                                                    np.max(clims()[0][int(np.floor(col % 4))]),2) # 5
                                                )
                        else:
                            cbar = plt.colorbar(inset_ax.images[0],location = 'bottom', cax = bar_ax[0], 
                                                format = FuncFormatter(fmt), 
                                                ticks = np.linspace(-3.14,3.14,2)
                                                ) 
                        
                        cbar.ax.get_xticklabels()[0].set_horizontalalignment('left')
                        cbar.ax.get_xticklabels()[1].set_horizontalalignment('right')
                        cbar.ax.tick_params(labelsize = 12)
                        cbar.set_label(names[int(np.floor(col % 4))],size=15)  # Add a label to the colorbar

                    inset_ax.axis("off")
            ax.axis("off")
            plt.close(voltage_and_switching_maps_fig)


    # Adjust the spacing between the plots as needed
    plt.tight_layout()

    # Show the layout
    plt.show()
    
    if filename is not None:
        visualizer.printer.savefig(
                    ax.figure, filename
                )

                                
                        
                        
            

def plot_figure_4(visualizer,filename):
    """
    Plots the figure 4 of the paper.
    """
    data, voltage = visualizer.get_hysteresis(scaled=True, loop_interpolated = True)
    data = torch.atleast_3d(torch.tensor(data.reshape(-1, 96)))

    data_names = ("LSQF", "NN")

    fig = plt.figure(figsize=(24, 24))


    # Define the GridSpec layout
    gs = GridSpec(60, 40, figure=fig)


    order = [['NN_fit_comp'],
            ['violin'],
            ['switching_maps']
            ]

    subplot_specs = [(0, 30, 0, 20 ), # top left: NN fit comparisons 
                    (0, 29, 20, 40), # g
                    (30, 46, 0, 60), #bottom 
                    ]
            
    
    for i, (r_start, r_end, c_start, c_end) in enumerate(subplot_specs):
        try: 
            idx = order[i]
        except: 
            break
        ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
        
        if idx[0] == 'NN_fit_comp':
            ax.axis("off")
            size=(1.25, 1.25)
            gaps=(1, 0.66)
            
            model, data = instantiate_hysteresis_model(visualizer, Train = False)
            
            fig_BMW = visualizer.hysteresis_comparison(data_names, nn_model=model, filename=None)

            
            #   list_ax.append(axs)
            axes_index = [5,4,3,2,1,0]

            
            for row in range(3):
                for col in range(2):
                    inset_ax = ax.inset_axes([(col/2)-col*0.186,1-(row+1)/3.2,1/3.2,1/3.2])
                                            
                    #inset_ax.set_xlim([-16,16])
                    inset_ax.set_xticks([-16,0,16])
                    inset_ax.get_xticklabels()[0].set_horizontalalignment('left')
                    inset_ax.get_xticklabels()[1].set_horizontalalignment('center')
                    inset_ax.get_xticklabels()[2].set_horizontalalignment('right')
                    
                    if row == 2: 
                        inset_ax.set_xlabel('Voltage(V)', fontsize=20)
                    else:
                        inset_ax.set_xlabel("")
                        inset_ax.xaxis.set_ticklabels([])
                    if row == 0: 
                        inset_ax.set_ylim([-1.7e-4,1.5e-4])
                        inset_ax.set_yticks(np.linspace(-1.55e-4,1.5e-4,5))
                        if col == 1:
                            inset_ax.set_yticklabels("")
                    elif row == 1: 
                        inset_ax.set_ylim([-1.55e-4,1.25e-4])
                        inset_ax.set_yticks(np.linspace(-1.5e-4,1e-4,6))
                        if col == 1:
                            inset_ax.set_yticklabels("")
                    else:
                        inset_ax.set_ylim([-6e-4, 1.1e-4])
                        if col == 1:
                            inset_ax.set_yticklabels("")
                    
                    
                        
                    for line in fig_BMW.get_axes()[axes_index[2*row+col]].get_lines():
                    #for line in ax_hyst_comp[2*row+col].get_lines():
                        label = line.get_label() if line.get_label() != '_nolegend_' else None
                        # Copying line properties like color, linestyle, marker, etc.
                        if col == 1:
                            inset_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                                    linestyle=line.get_linestyle(), marker=line.get_marker(), label=label)
                        else:
                            inset_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                                    linestyle=line.get_linestyle(), marker=line.get_marker())
                        
                    if col==0:   
                        set_sci_notation_label(
                            inset_ax, corner="top left", axis="y", stroke_color="w", linewidth=0.5,
                            textsize = 15, offset_points = (0,30)
                        )
                    
                    
                    
                    inset_ax.tick_params(axis='x',labelsize=20)
                    inset_ax.tick_params(axis='y',labelsize=20)
                                
                    plt.tight_layout()    
                    
                    
                    
                    if row == 0:
                    
                        labelfigs(inset_ax,
                                string_add="Best",
                                loc ='ct',
                                label_size=17,
                                inset_fraction=(0.075,0.5),
                                style = 'b',
                                horizontalalignment = "center"
                                )
                        
                    # get legend handles and their corresponding labels
                        handles, labels = inset_ax.get_legend_handles_labels()
                        

                        inset_ax.legend(handles,labels, loc=(1.05,0.65),fontsize = 17)
                        

                    elif row == 1:
                        labelfigs(inset_ax,
                                string_add="Median",
                                loc ='ct',
                                label_size=17,
                                inset_fraction=(0.075,0.5),
                                style = 'b',
                                horizontalalignment = "center"
                                )
                        
                    elif row == 2:
                        labelfigs(inset_ax,
                                string_add="Worst",
                                loc ='ct',
                                label_size=17,
                                inset_fraction=(0.075,0.5),
                                style = 'b',
                                horizontalalignment="center"
                                )
                        ax.set_xticks([1.2,1.3,1.4])
                        
                    
                    labelfigs(inset_ax,
                        number=2*row+col,
                        loc ='tr',
                        label_size=17,
                        inset_fraction=(0.075,0.075),
                        style = 'b'
                        )
            plt.close(fig_BMW)
        elif idx[0] == 'violin':
            visualizer.violin_plot_comparison_hysteresis(model,
                                        torch.atleast_3d(torch.tensor(data.reshape(-1, 96))),
                                        filename=None,ax=ax) 

            # labels the figure and does some styling
            labelfigs(ax, string_add = 'g', loc ='tr',label_size=20, style="b", inset_fraction=(0.05,0.05))
            ax.set_ylabel("Scaled Hysteresis Results",fontsize=25)
            ax.set_xlabel("")
            
            ax.tick_params(axis='x',labelsize=20)
            ax.tick_params(axis='y',labelsize=20)

            # Get the legend associated with the plot
            legend = ax.get_legend()
            legend.set_title("")
            plt.setp(legend.get_texts(), fontsize=20) # Set the label size
        
        elif idx[0] == "switching_maps": 
            
            NN_params = instantiate_hysteresis_model_params(visualizer,model,data)
                
            fig_hysteresis = visualizer.hysteresis_maps(NN_params, cycle=0, filename=None);
            fig_scalar = FigDimConverter((1/2.5, 1/2.5))

            
            for row in range(2):
                for col in range(9):
                    inset_ax = ax.inset_axes([-0.153+(col/8.99),1-(row+1.2)/2.4-row/12,1/2.4,1/2.4])
                    inset_ax.imshow(fig_hysteresis.get_axes()[col+9*row].get_images()[0].get_array().data,cmap = 'viridis',
                                        vmin = visualizer.hysteresis_maps_clims[col][0], vmax = visualizer.hysteresis_maps_clims[col][1])
                    inset_ax.axis("off")
                    
                    if row == 1:
                        bar_ax = []
                        
                        pos_inch = [(col/22.5), -0.008, 1/23, 1/300  ] #fills axes
                        bar_ax.append(ax.inset_axes(fig_scalar.to_relative(pos_inch)))

                        cbar = plt.colorbar(inset_ax.images[0],      
                                            cax=bar_ax[0], format=FuncFormatter(visualizer.hysteresis_maps_fmt),orientation = 'horizontal',
                                            ticks = [visualizer.hysteresis_maps_clims[col][0], visualizer.hysteresis_maps_clims[col][1]])
                        
                        cbar.ax.get_xticklabels()[0].set_horizontalalignment('left')
                        cbar.ax.get_xticklabels()[1].set_horizontalalignment('right')

                        cbar.ax.tick_params(labelsize = 11)

                        
                        cbar.set_label(visualizer.hysteresis_maps_colorbar_labels[col],size=15,loc='center')  # Add a label to the colorbar
                        
                            
                    
            labelfigs(ax,
                    string_add="Least Squares Fit Method",
                    loc='ct',label_size=25,inset_fraction = (0.05,0.5),style='b',
                    horizontalalignment = 'center',verticalalignment='center')
            labelfigs(ax,
                    string_add="h",
                    loc='tl',label_size=25,inset_fraction = (0.3,-0.015),style='b',
                    horizontalalignment = 'center',verticalalignment='center')
        
            labelfigs(ax,
                    string_add="Neural Network with Trust Region CG",
                    loc='ct',label_size=25,inset_fraction = (0.55,0.5),style='b',
                    horizontalalignment = 'center',verticalalignment='center')
            labelfigs(ax,
                    string_add="i",
                    loc='tl',label_size=25,inset_fraction = (0.8,-0.015),style='b',
                    horizontalalignment = 'center',verticalalignment='center')
                        
            ax.axis("off")         
            plt.close(fig_hysteresis)

    # Adjust the spacing between the plots as needed
    plt.tight_layout()
                    
    # Show the layout
    plt.show()

    if filename is not None:
        visualizer.printer.savefig(
                    ax.figure, filename
                )

# TODO: remove the y-axis ticks for the violin plots for noise levels 2,7
def plot_figure_5(visualizer,
                    filename):
    """
    Plots the figure 5 of the paper.
    """
    
    fig = plt.figure(figsize=(24, 24))


    # Define the GridSpec layout
    gs = GridSpec(60, 45, figure=fig)


    order = [['violin_noise_0'],
            ['switching_maps_noise_0'],
            ['violin_noise_2'],
            ['switching_maps_noise_2'],
            ['switching_maps_noise_4'],
            ['violin_noise_7'],
            ['switching_maps_noise_7']
            ]

    subplot_specs = [(0, 17, 0, 13 ), # top left: violin noise level 0 
                    (20, 26, 0, 45), # switching maps:noise = 0             
                    (0, 17, 16, 29), # top middle: violin noise level 2
                    (27, 33, 0, 45), # switching maps:noise = 2
                    (34, 40, 0, 45), # switching maps:noise = 4
                    (0, 17, 32, 45), # top right, violin noise level 7
                    (41, 47, 0, 45), # switching maps:noise = 7
                    ]

    
    violin_plot_noise_to_figlabel = {
        'violin_noise_0': 'a',
        'violin_noise_2': 'b',
        'violin_noise_7': 'c',
    }
    


    for i, (r_start, r_end, c_start, c_end) in enumerate(subplot_specs):
        ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
        
        idx = order[i]
        if idx[0].startswith('violin_noise'):
                # this print statement is a low-tech way to track progress
                # since each violin plot takes a while (~40 seconds) to render
                
                
                
                
            
                # visualizer.noise = int(idx[0][-1])
                # visualizer.get_dataset(noise = visualizer.noise)

                state_ = {'resampled': True,
                    'raw_format': 'complex',
                    'fitter': 'LSQF',
                    'scaled': True,
                    'output_shape': 'index',
                    'measurement_state': 'all',
                    'resampled_bins': 165,
                    'LSQF_phase_shift': np.pi/2, #1.5707963267948966,
                    'NN_phase_shift': np.pi/2,
                    'noise': int(idx[0][-1])}

                visualizer.set_attributes(**state_)
                print(f"instantiating model for noise level {idx[0][-1]} ...")

                model = instantiate_SHO_model(visualizer,
                    noise = int(idx[0][-1]),
                    model_basename = f"SHO_Fitter_original_data_noise_{idx[0][-1]}",
                    datafed_path = '2024_SHO_Fitting/Noisy_NN',
                    script_path = './Paper_Figures.ipynb',
                    seed=42, 
                    device = 'cuda:0'
                    )
                
                X_data, NN_params = instantiate_SHO_model_params(visualizer,model)
                
                print(f"working on the violin plot for noise level {idx[0][-1]} ...")
                
                visualizer.violin_plot_comparison_SHO(
                        state_,
                        model,
                        X_data,
                        NN_params,
                        filename=None,
                        label="NN",
                        ax=ax,
                        figlabel=violin_plot_noise_to_figlabel[idx[0]],
                        label_size=20,
                        loc = 'tr',
                        inset_fraction = (0.075,0.075)
                    )
                if idx[0] == 'violin_noise_0':
                    ax.set_ylabel("Scaled SHO Results",fontsize=20)
                    
                        # Get the legend associated with the plot
                    legend = ax.get_legend()
                    legend.set_title("")
                    plt.setp(legend.get_texts(), fontsize=20) # Set the label size
                
                else:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                    ax.get_legend().remove()
                
                ax.set_xlabel("")
                
                ax.set_title(f"Noise Level {idx[0][-1]}",fontsize=20)
                ax.tick_params(axis='x',labelsize=20)
                ax.tick_params(axis='y',labelsize=20)
                ax.set_yticks(np.linspace(-8,8,9))
                
                
        
        elif idx[0].startswith('switching_maps_noise'):
                if idx[0][-1] == "4":
                    print(f"instantiating model for noise level {idx[0][-1]} ...")

                    model = instantiate_SHO_model(visualizer,
                        noise = int(idx[0][-1]),
                        model_basename = f"SHO_Fitter_original_data_noise_{idx[0][-1]}",
                        datafed_path = '2024_SHO_Fitting/Noisy_NN',
                        script_path = './Paper_Figures.ipynb',
                        seed=42, 
                        device = 'cuda:0'
                        )
                    
                    X_data, NN_params = instantiate_SHO_model_params(visualizer,model)

                
                # visualizer.noise = int(idx[0][-1])
                # visualizer.get_dataset(noise = visualizer.noise)
                
                if visualizer.noise == 0:
                    labelfigs(ax, string_add = "d", inset_fraction = (-0.1, 0.010),label_size=20,style='b')
                    labelfigs(ax, string_add = "\u25CF", inset_fraction = (-0.1, 0.085),label_size=20,style='b')

                    labelfigs(ax, string_add = "e", inset_fraction = (-0.1, 0.215),label_size=20,style='b')
                    labelfigs(ax, string_add = "\u25BC", inset_fraction = (-0.1, 0.2915),label_size=20,style='b')

                    labelfigs(ax, string_add = "f", inset_fraction = (-0.1, 0.424),label_size=20,style='b')
                    labelfigs(ax, string_add = "\u25B2", inset_fraction = (-0.1, 0.496),label_size=20,style='b')

                    labelfigs(ax, string_add = "g", inset_fraction = (-0.1, 0.63),label_size=20,style='b')
                    labelfigs(ax, string_add = "\u25BA", inset_fraction = (-0.1, 0.705),label_size=20,style='b')


                    labelfigs(ax, string_add = "h", inset_fraction = (-0.1, 0.835),label_size=20,style='b')
                    labelfigs(ax, string_add = "\u25C0", inset_fraction = (-0.1, 0.91),label_size=20,style='b')

                # LSQF_ = {'resampled': True,
                #     'raw_format': 'complex',
                #     'fitter': 'LSQF',
                #     'scaled': False,
                #     'output_shape': 'index',
                #     'measurement_state': 'all',
                #     'resampled_bins': 165,
                #     'LSQF_phase_shift': 1.5707963267948966,
                #     'NN_phase_shift': 1.5707963267948966,
                #     'noise': int(idx[0][-1])}
                
                #LSQF_Params = visualizer.SHO_fit_results(state = LSQF_)
                LSQF_Params = visualizer.SHO_fit_results(state = state_)

                voltage_and_switching_maps_fig = visualizer.SHO_switching_maps(
                    SHO_ = [LSQF_Params,NN_params],
                    labels = ["LSQF", "NN"], 
                    filename=None,
                    colorbars=False,
                    )

                ax2 = voltage_and_switching_maps_fig.axes[1:]
                fig_scalar = FigDimConverter((1/6, 1/6))
                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_color(None)
                ax.spines["right"].set_visible(False)
                ax.set_ylabel(f"NN Noise {visualizer.noise}   LSQF Noise {visualizer.noise}",fontsize=10)
                ax.yaxis.set_label_coords(-0.015, 0.5)
                    
                for row in range(2):
                    for col in range(20):
                        if col > 11:
                            col_used = col + 12
                        else:
                            col_used = col
                        
                        
                        inset_ax = ax.inset_axes([-0.24+(col/20)+np.floor(col/4)/175,1-(row+1)/2-row/50,1/2,1/2])
                        inset_ax.imshow(ax2[(12*row)+col_used].get_images()[0].get_array().data, clim = clims()[0][int(np.floor(col % 4))])
                        inset_ax.axis("off")


                        if row == 1 and visualizer.noise == 7:
                            bar_ax = []
                            pos_inch = [-0.0024+(col/120)+np.floor(col/4)/1050, -0.01, 1/124, 1/200  ] #fills axes
                            bar_ax.append(ax.inset_axes(fig_scalar.to_relative(pos_inch)))

                            if int(np.floor(col % 4) in [0,2]):
                                cbar = plt.colorbar(inset_ax.images[0],          #axs[1,col].images[0],
                                                cax=bar_ax[0], format=FuncFormatter(fmt),orientation = 'horizontal',
                                                ticks = [clims()[0][int(np.floor(col % 4))][0], clims()[0][int(np.floor(col % 4))][1]])
                            elif int(np.floor(col % 4) == 1):
                                cbar = plt.colorbar(inset_ax.images[0],          #axs[1,col].images[0],
                                                cax=bar_ax[0], format=FuncFormatter(fmt_resonance),orientation = 'horizontal',
                                                ticks = [clims()[0][int(np.floor(col % 4))][0], clims()[0][int(np.floor(col % 4))][1]])
                            else:
                                cbar = plt.colorbar(inset_ax.images[0],          #axs[1,col].images[0],
                                                cax=bar_ax[0], format=FuncFormatter(fmt),orientation = 'horizontal',
                                                ticks = [-3.14,3.14])    
                            
                            
                            cbar.ax.get_xticklabels()[0].set_horizontalalignment('left')
                            cbar.ax.get_xticklabels()[1].set_horizontalalignment('right')

                            cbar.ax.tick_params(labelsize = 7)

                plt.close(voltage_and_switching_maps_fig)
    plt.tight_layout()


    # Show the layout
    plt.show()
    if filename is not None:
        visualizer.printer.savefig(
                    ax.figure, filename
                )