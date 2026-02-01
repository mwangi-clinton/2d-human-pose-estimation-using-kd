"""
Tool to compute Centered Kernel Alignment (CKA) in PyTorch w/ GPU (single or multi).
Modified to support models requiring dataset_index parameter.

Repo: https://github.com/numpee/CKA.pytorch
Author: Dongwan Kim (Github: Numpee)
Year: 2022
Modified: Added dataset_index support
"""

from __future__ import annotations

from typing import Tuple, Optional, Callable, Type, Union, TYPE_CHECKING, List

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from .hook_manager import HookManager, _HOOK_LAYER_TYPES
from .metrics import AccumTensor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CKACalculator:
    def __init__(self, model1: nn.Module, model2: nn.Module, dataloader: DataLoader,
                 hook_fn: Optional[Union[str, Callable]] = None,
                 hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES, 
                 num_epochs: int = 10,
                 group_size: int = 512, 
                 epsilon: float = 1e-4, 
                 is_main_process: bool = True,
                 dataset_index: Optional[int] = None,
                 model1_requires_dataset_index: bool = False,
                 model2_requires_dataset_index: bool = False,
                 debug: bool = False) -> None:
        """
        Class to extract intermediate features and calculate CKA Matrix.
        
        :param model1: model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param model2: second model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param dataloader: Torch DataLoader for dataloading. Assumes first return value contains input images.
        :param hook_fn: Optional - Hook function or hook name string for the HookManager. Options: [flatten, avgpool]. Default: flatten
        :param hook_layer_types: Types of layers (modules) to add hooks to.
        :param num_epochs: Number of epochs for cka_batch. Default: 10
        :param group_size: group_size for GPU acceleration. Default: 512
        :param epsilon: Small multiplicative value for HSIC. Default: 1e-4
        :param is_main_process: is current instance main process. Default: True
        :param dataset_index: Dataset index to pass to models that require it (e.g., VitPose with multiple experts)
        :param model1_requires_dataset_index: Whether model1 requires dataset_index parameter
        :param model2_requires_dataset_index: Whether model2 requires dataset_index parameter
        :param debug: Enable debug printing
        """
        self.model1 = model1
        self.model2 = model2
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.epsilon = epsilon
        self.is_main_process = is_main_process
        self.dataset_index = dataset_index
        self.model1_requires_dataset_index = model1_requires_dataset_index
        self.model2_requires_dataset_index = model2_requires_dataset_index
        self.debug = debug

        self.model1.eval()
        self.model2.eval()
        self.hook_manager1 = HookManager(self.model1, hook_fn, hook_layer_types, calculate_gram=True)
        self.hook_manager2 = HookManager(self.model2, hook_fn, hook_layer_types, calculate_gram=True)
        self.module_names_X = None
        self.module_names_Y = None
        self.num_layers_X = None
        self.num_layers_Y = None
        self.num_elements = None

        # Metrics to track
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None
        
        if self.debug:
            print(f"CKACalculator initialized:")
            print(f"  - Model1 requires dataset_index: {model1_requires_dataset_index}")
            print(f"  - Model2 requires dataset_index: {model2_requires_dataset_index}")
            print(f"  - Dataset index: {dataset_index}")

    def _forward_model(self, model, imgs, requires_dataset_index):
        """Helper method to forward pass through model with optional dataset_index"""
        if requires_dataset_index:
            if self.dataset_index is None:
                raise ValueError(
                    f"Model requires dataset_index but none was provided. "
                    f"Please pass dataset_index to CKACalculator.__init__()"
                )
            if self.debug:
                print(f"  Forwarding with dataset_index={self.dataset_index}")
            return model(imgs, dataset_index=self.dataset_index)
        else:
            if self.debug:
                print(f"  Forwarding without dataset_index")
            return model(imgs)

    @torch.no_grad()
    def calculate_cka_matrix(self) -> torch.Tensor:
        curr_hsic_matrix = None
        curr_self_hsic_x = None
        curr_self_hsic_y = None
        
        for epoch in range(self.num_epochs):
            loader = tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self.is_main_process)
            for it, (imgs, *_) in enumerate(loader):
                imgs = imgs.cuda(non_blocking=True)
                
                if self.debug and it == 0:
                    print(f"\nEpoch {epoch}, Batch {it}:")
                    print(f"  Input shape: {imgs.shape}")
                
                # Forward pass with optional dataset_index
                try:
                    self._forward_model(self.model1, imgs, self.model1_requires_dataset_index)
                except Exception as e:
                    print(f"Error in model1 forward pass: {e}")
                    raise
                
                try:
                    self._forward_model(self.model2, imgs, self.model2_requires_dataset_index)
                except Exception as e:
                    print(f"Error in model2 forward pass: {e}")
                    raise
                
                all_layer_X, all_layer_Y = self.extract_layer_list_from_hook_manager()

                # Initialize values on first loop
                if self.num_layers_X is None:
                    curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y = self._init_values(all_layer_X, all_layer_Y)
                    if self.debug:
                        print(f"  Initialized with {self.num_layers_X} layers in model1, {self.num_layers_Y} layers in model2")

                # Get self HSIC values --> HSIC(K, K), HSIC(L, L)
                self._calculate_self_hsic(all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y)

                # Get cross HSIC values --> HSIC(K, L)
                self._calculate_cross_hsic(all_layer_X, all_layer_Y, curr_hsic_matrix)

                self.hook_manager1.clear_features()
                self.hook_manager2.clear_features()
                curr_hsic_matrix.fill_(0)
                curr_self_hsic_x.fill_(0)
                curr_self_hsic_y.fill_(0)

        # Update values across GPUs
        hsic_matrix = self.hsic_matrix.compute()
        hsic_x = self.self_hsic_x.compute()
        hsic_y = self.self_hsic_y.compute()
        self.cka_matrix = hsic_matrix.reshape(self.num_layers_Y, self.num_layers_X) / torch.sqrt(hsic_x * hsic_y)
        
        if self.debug:
            print(f"\nFinal CKA matrix shape: {self.cka_matrix.shape}")
            print(f"CKA diagonal: {self.cka_matrix.diagonal()}")
        
        return self.cka_matrix

    def extract_layer_list_from_hook_manager(self) -> Tuple[List, List]:
        all_layer_X, all_layer_Y = self.hook_manager1.get_features(), self.hook_manager2.get_features()
        return all_layer_X, all_layer_Y

    def hsic1(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Batched version of HSIC.
        :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :return: HSIC tensor, Size = (B)
        '''
        assert K.size() == L.size()
        assert K.dim() == 3
        K = K.clone()
        L = L.clone()
        n = K.size(1)

        # K, L --> K~, L~ by setting diagonals to zero
        K.diagonal(dim1=-1, dim2=-2).fill_(0)
        L.diagonal(dim1=-1, dim2=-2).fill_(0)

        KL = torch.bmm(K, L)
        trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
        middle_term = K.sum((-1, -2), keepdim=True) * L.sum((-1, -2), keepdim=True)
        middle_term /= (n - 1) * (n - 2)
        right_term = KL.sum((-1, -2), keepdim=True)
        right_term *= 2 / (n - 2)
        main_term = trace_KL + middle_term - right_term
        hsic = main_term / (n ** 2 - 3 * n)
        return hsic.squeeze(-1).squeeze(-1)

    def reset(self) -> None:
        # Set values to none, clear feature and hooks
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None
        self.hook_manager1.clear_all()
        self.hook_manager2.clear_all()

    def _init_values(self, all_layer_X, all_layer_Y):
        self.num_layers_X = len(all_layer_X)
        self.num_layers_Y = len(all_layer_Y)
        self.module_names_X = self.hook_manager1.get_module_names()
        self.module_names_Y = self.hook_manager2.get_module_names()
        self.num_elements = self.num_layers_Y * self.num_layers_X
        curr_hsic_matrix = torch.zeros(self.num_elements).cuda()
        curr_self_hsic_x = torch.zeros(1, self.num_layers_X).cuda()
        curr_self_hsic_y = torch.zeros(self.num_layers_Y, 1).cuda()
        self.hsic_matrix = AccumTensor(torch.zeros_like(curr_hsic_matrix)).cuda()
        self.self_hsic_x = AccumTensor(torch.zeros_like(curr_self_hsic_x)).cuda()
        self.self_hsic_y = AccumTensor(torch.zeros_like(curr_self_hsic_y)).cuda()
        return curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y

    def _calculate_self_hsic(self, all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y):
        for start_idx in range(0, self.num_layers_X, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_X)
            K = torch.stack([all_layer_X[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_x[0, start_idx:end_idx] += self.hsic1(K, K) * self.epsilon
        for start_idx in range(0, self.num_layers_Y, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_Y)
            L = torch.stack([all_layer_Y[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_y[start_idx:end_idx, 0] += self.hsic1(L, L) * self.epsilon

        self.self_hsic_x.update(curr_self_hsic_x)
        self.self_hsic_y.update(curr_self_hsic_y)

    def _calculate_cross_hsic(self, all_layer_X, all_layer_Y, curr_hsic_matrix):
        for start_idx in range(0, self.num_elements, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_elements)
            K = torch.stack([all_layer_X[i % self.num_layers_X] for i in range(start_idx, end_idx)], dim=0)
            L = torch.stack([all_layer_Y[j // self.num_layers_X] for j in range(start_idx, end_idx)], dim=0)
            curr_hsic_matrix[start_idx:end_idx] += self.hsic1(K, L) * self.epsilon
        self.hsic_matrix.update(curr_hsic_matrix)


def gram(x: torch.Tensor) -> torch.Tensor:
    return x.matmul(x.t())