def figure_format(fig_width, fig_height=None):
    """
    Parameter
    ---------
    fig_width : float
        figure width in inches
        e.g. 3.4

    fig_height : float
        figure height in inches
        e.g. 3.4

    Returns
    -------
    A tuple with:
    - Float corresponding to figure width and height
    - A dictionary containing several parameters, such as fontsize
      , etc. If fig_width is not set, the default value is 3.4 inch,
      corresponding to the width of a column in a double colomn paper.
    """
    golden_ratio = 1.618  # Aesthetic ratio
    if fig_height is None:
        fig_height = fig_width / golden_ratio

    fig_size = [fig_width, fig_height]
    fontsize = 7
    linewidth = 0.5
    params = {
        "backend": "ps",
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "axes.linewidth": linewidth,
        "font.size": fontsize,
        "font.family": "serif",
        "legend.frameon": False,
        "legend.fontsize": fontsize,
        "legend.loc": "best",
        "lines.linewidth": linewidth,
        #
        "xtick.labelsize": fontsize,
        "xtick.direction": "in",
        "xtick.top": True,
        "xtick.major.size": 2.0,
        "xtick.major.top": True,
        "xtick.major.width": 0.3,
        "xtick.minor.size": 1.0,
        "xtick.minor.top": True,
        "xtick.minor.width": 0.3,
        #
        "ytick.labelsize": fontsize,  
        "ytick.direction": "in",        
        "ytick.right": True,
        "ytick.major.size": 2.0,
        "ytick.major.width": 0.3,
        "ytick.minor.size": 1.0,
        "ytick.minor.width": 0.3,
        "figure.figsize": fig_size,
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "ps.usedistiller": "xpdf",
    }
    return (fig_width, fig_height, params)
