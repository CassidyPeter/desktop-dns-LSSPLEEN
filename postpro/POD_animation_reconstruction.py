import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

def POD_animation_reconstruction(
        casename, FPS, DPI, frames_to_render, dt,
        reconstructions, u_normalisation, component):

    """
    Creates animation of POD reconstruction of u' or v' depending on 'component'.
    component must be 'Up' or 'Vp' (case insensitive).
    """

    # ------------------------ Component Selection -----------------------------

    component = component.capitalize()
    if component not in ["Up", "Vp"]:
        raise ValueError("component must be 'Up' or 'Vp' (Phi_w_norm does not exist)")

    # Map names
    comp_letter = component[0].lower()     # 'u' or 'v'
    comp_label  = f"${comp_letter}'$"      # Latex for colorbar and title

    # --------------------------------------------------------------------------

    plt.rcParams['font.size'] = '16'

    print("Loading data from 'animation_data.npz'...")
    try:
        data = np.load(os.path.join(casename, 'animation_data.npz'))

        # Raw fluctuation fields
        Up = data['Up']
        Vp = data['Vp']

        # Select correct field
        field_map = {"Up": Up, "Vp": Vp}
        F = field_map[component]

        # POD basis selection
        Phi_map = {
            "Up": data['Phi_u_norm'],
            "Vp": data['Phi_v_norm']
        }
        Phi_norm = Phi_map[component]

        a_t = data['a_t']

        x_plot = data['x_plot']
        y_plot = data['y_plot']
        Nx = data['Nx'].item()
        Ny = data['Ny'].item()
        Nt = F.shape[1]

        print(f"...Data loaded. Found {Nt} snapshots.")

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()

    # ------------------------ Frame Count -------------------------------------

    if frames_to_render == -1:
        frames_to_render = Nt

    output_filename = os.path.join(
        casename, "Figures",
        f"POD_reconstruction_{component}_animation_{frames_to_render}Nt.gif"
    )

    os.makedirs(os.path.join(casename, 'Figures'), exist_ok=True)

    # ------------------------ Plot Layout -------------------------------------

    clim = [-1, 1]
    print("Setting up plot...")

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(20, 4),
        sharex=True, sharey=True
    )
    plt.set_cmap('seismic')

    im_orig = None
    im_recons = []
    title_orig = None

    # ------------------- Initial Frame ----------------------------------------

    ax = axes.flat[0]
    f_initial = F[:, 0].reshape(Nx, Ny, order='F')
    im_orig = ax.pcolormesh(
        x_plot, y_plot, f_initial,
        shading='gouraud', vmin=clim[0], vmax=clim[1]
    )
    title_orig = ax.set_title(f'Original {component} field (t=0)')
    ax.set_ylabel('Wall normal Distance y/Cax')

    # ------------------ Reconstructions (initial frame) -----------------------

    for i, n_modes in enumerate(reconstructions):
        ax = axes.flat[i+1]

        f_recon = np.tensordot(
            Phi_norm[:, :, :n_modes],
            a_t[0, :n_modes],
            axes=([2], [0])
        )

        f_vis = f_recon / np.max(np.abs(f_recon)) * u_normalisation

        im = ax.pcolormesh(
            x_plot, y_plot, f_vis,
            shading='gouraud', vmin=clim[0], vmax=clim[1]
        )
        im_recons.append(im)

        ax.set_title(f"First {n_modes} modes")

        if i in [1, 2]:
            ax.set_xlabel('s/SL')

    ax.axis([0.4, 0.95, 0, 0.003])
    plt.tight_layout()

    # ------------------------ Colorbar ----------------------------------------

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im_orig, cax=cbar_ax)
    cbar.ax.set_title(comp_label, pad=10)

    # ------------------------ Animation Update ---------------------------------

    def animate(frame):

        # Update original field
        f_frame = F[:, frame].reshape(Nx, Ny, order='F')
        im_orig.set_array(f_frame.ravel())
        title_orig.set_text(f'Original {component} field (t={round(frame*dt, 4)} s)')

        artists = [im_orig, title_orig]

        # Update reconstructions
        for i, n_modes in enumerate(reconstructions):
            f_recon = np.tensordot(
                Phi_norm[:, :, :n_modes],
                a_t[frame, :n_modes],
                axes=([2], [0])
            )
            f_vis = f_recon / np.max(np.abs(f_recon)) * u_normalisation

            im_recons[i].set_array(f_vis.ravel())
            artists.append(im_recons[i])

        return artists

    # ------------------------ Save Animation -----------------------------------

    print(f"Generating animation ({frames_to_render} frames)...")
    anim = animation.FuncAnimation(fig, animate,
                                   frames=frames_to_render,
                                   blit=True)

    print(f"Saving animation to '{output_filename}'...")
    try:
        anim.save(output_filename, fps=FPS, dpi=DPI, writer='pillow')
        print(f"\nAnimation saved successfully to '{output_filename}'!\n")
    except Exception as e:
        print(f"Error saving animation: {e}")
