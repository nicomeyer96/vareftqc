import matplotlib.pyplot as plt


WIDTHS = {
    'onecolumn': {
        'letter': 7.08
    },
    'twocolumn': {
        'letter': 3.41
    }
}

FONTSIZES = {
    10: {
        'tiny': 5,
        'scriptsize': 7,
        'footnotesize': 8,
        'small': 9,
        'normalsize': 10,
        'large': 12,
        'Large': 14,
        'LARGE': 17,
        'huge': 20,
        'Huge': 25
    },
    11: {
        'tiny': 6,
        'scriptsize': 8,
        'footnotesize': 9,
        'small': 10,
        'normalsize': 11,
        'large': 12,
        'Large': 14,
        'LARGE': 17,
        'huge': 20,
        'Huge': 25
    },
    12: {
        'tiny': 6,
        'scriptsize': 8,
        'footnotesize': 10,
        'small': 11,
        'normalsize': 12,
        'large': 14,
        'Large': 17,
        'LARGE': 20,
        'huge': 25,
        'Huge': 25
    }
}



def setup_figure_latex_layout(aspect_ratio=1/1.62, width_ratio=1.0, columns='twocolumn', paper='letter', fontsize=11):
    plt.rcdefaults()
    plt.style.use('default')

    # Use the default fontsize scaling of LaTeX
    fontsizes = FONTSIZES[fontsize]

    tex_fonts = {
                # Use LaTeX to write all text, load fonts
                "text.usetex": True,
                'text.latex.preamble': r"\usepackage{amsmath}",
                "font.family": "serif",
                "font.serif": "STIX",
                "mathtext.fontset": 'stix',
                # Set font sizes
                "axes.labelsize": fontsizes['small'],
                "axes.titlesize": fontsizes['large'],
                "font.size": fontsizes['small'],
                "legend.fontsize": fontsizes['footnotesize'],
                "xtick.labelsize": fontsizes['footnotesize'],
                "ytick.labelsize": fontsizes['footnotesize'],
                "figure.labelsize": fontsizes['small'],
                'xtick.major.size': 3,
                'xtick.major.width': .5,
                'ytick.major.size': 3,
                'ytick.major.width': .5,
            }

    plt.rcParams.update(tex_fonts)
    plt.rc('text.latex', preamble=r'\usepackage{physics,graphicx,amssymb,MnSymbol}')

    width = WIDTHS[columns][paper] * width_ratio
    height = width * aspect_ratio
    return width, height
