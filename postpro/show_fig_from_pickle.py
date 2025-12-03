def show_fig_from_pickle(pickle_file):
    import pickle
    import matplotlib.pyplot as plt
    from IPython.display import display

    # Load figure
    with open(pickle_file, 'rb') as f:
        fig = pickle.load(f)

    # Display in Jupyter
    display(fig)

    return fig
