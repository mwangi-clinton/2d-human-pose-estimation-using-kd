import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import Optional

def plot_cka_heatmap(
    cka_matrix: torch.Tensor,
    model1_name: str,
    model2_name: str,
    title: str = "CKA Heatmap",
    figsize: tuple = (7, 7),
    cmap: str = 'inferno',
    save_path: Optional[str] = None
):
    """
    Plots a heatmap of the CKA matrix.

    Args:
        cka_matrix: Tensor of shape (num_layers_y, num_layers_x)
        model1_name: Name of the first model (X-axis)
        model2_name: Name of the second model (Y-axis)
        title: Title of the plot
        figsize: Tuple for figure size
        cmap: Colormap to use
        save_path: If provided, saves the plot to this path
    """
    if isinstance(cka_matrix, torch.Tensor):
        cka_matrix = cka_matrix.cpu().numpy()
        
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cka_matrix,
        xticklabels=False,
        yticklabels=False,
        cmap=cmap,
        ax=ax
    )
    
    ax.set_xlabel(model1_name, fontsize=12)
    ax.set_ylabel(model2_name, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()  # Match standard matrix visualization origin (top-left vs bottom-left)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved CKA heatmap to {save_path}")
        
    plt.show()
    return fig, ax
