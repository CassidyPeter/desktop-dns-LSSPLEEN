import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import pandas as pd

def POD_animation_prime(casename, FPS, DPI, frames_to_render, dt, component):
    """
    Creates animation of u', v', or w' depending on 'component'.
    component must be one of: 'Up', 'Vp', 'Wp' (case insensitive)
    """

    component = component.capitalize()   # Normalize ("up" → "Up")
    if component not in ["Up", "Vp", "Wp"]:
        raise ValueError("component must be one of: 'Up', 'Vp', 'Wp'")

    plt.rcParams['font.size'] = '16'

    print("Loading data from 'animation_data.npz'...")
    try:
        data = np.load(os.path.join(casename, 'animation_data.npz'))

        Up = data['Up']
        Vp = data['Vp']
        Wp = data['Wp']
        x_plot = data['x_plot']
        y_plot = data['y_plot']
        Nx = data['Nx'].item()
        Ny = data['Ny'].item()

        # Select correct field
        field_map = {"Up": Up, "Vp": Vp, "Wp": Wp}
        F = field_map[component]

        Nt = F.shape[1]
        print(f"...Data loaded. Found {Nt} snapshots.")

    except FileNotFoundError:
        print("Error: 'animation_data.npz' not found.")
        sys.exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()

    # Load transition/separation lines
    sep_trans = {"trans_ss_grad":0, "trans_ss_midp":0, "trans_ps_grad":0,
                 "trans_ps_midp":0, "sep_ss":0, "reatt_ss":0,
                 "sep_ps":0, "reatt_ps":0}

    for key in sep_trans:
        sep_trans[key] = pd.read_csv(os.path.join(casename, key + '.csv'))['s/SL'].item()

    if frames_to_render == -1:
        frames_to_render = Nt

    output_filename = os.path.join(
        casename, 'Figures', f'{component}_animation_{frames_to_render}Nt.gif')

    print("Setting up plot...")
    fig, ax = plt.subplots(figsize=(20, 4))
    plt.set_cmap('seismic')

    print("Calculating color limits...")
    sample_data = F[:, ::max(1, Nt // 100)].ravel()
    vmin, vmax = np.percentile(sample_data, [2.5, 97.5])
    # Or fixed:
    vmin, vmax = -1, 1
    print(f"Color limits set to: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Initial frame
    f0 = F[:, 0].reshape(Nx, Ny, order='F')

    im = ax.pcolormesh(x_plot, y_plot, f0,
                       shading='gouraud',
                       vmin=vmin,
                       vmax=vmax)

    # Separation/reattachment/trans lines
    if sep_trans['sep_ss'] > 0:
        ax.axvline(sep_trans['sep_ss'], color='k', linestyle='dotted', linewidth=1)
    ax.axvline(sep_trans['trans_ss_grad'], color='k', linestyle='dashed', linewidth=1)
    if sep_trans['reatt_ss'] > 0:
        ax.axvline(sep_trans['reatt_ss'], color='k', linestyle='dashdot', linewidth=1)

    cbar = fig.colorbar(im, ax=ax, location='right')
    cbar.ax.set_title(f"${component[0].lower()}'$ [m/s]", pad=10)

    ax.set_ylabel('Wall normal Distance y/Cax')
    ax.set_xlabel('s/SL')

    title = ax.set_title(f"{component} (t=0)")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.axis([0.4, 0.95, 0, 0.003])

    # Animation update
    def animate(frame):
        f_frame = F[:, frame].reshape(Nx, Ny, order='F')
        im.set_array(f_frame.ravel())
        title.set_text(f"{component} (t={round(frame*dt,4)} s)")
        return im, title

    print(f"Generating animation ({frames_to_render} frames)...")
    anim = animation.FuncAnimation(fig, animate,
                                   frames=frames_to_render,
                                   blit=True)

    print(f"Saving animation to '{output_filename}'...")
    try:
        anim.save(output_filename,
                  fps=FPS,
                  dpi=DPI,
                  writer='pillow',
                  savefig_kwargs={"transparent": True})
        print(f"\nAnimation saved successfully to '{output_filename}'!")
    except Exception as e:
        print(f"\nError saving animation: {e}")

