import functools
import inspect
from collections.abc import Iterable
from typing import Callable, Dict, Union, Any
import torch
import numpy as np
from pennylane import QNode
import pennylane as qml
try:
    import torch
    from torch.nn import Module

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the TorchLayer.
    from unittest.mock import Mock

    Module = Mock
    TORCH_IMPORTED = False


class TorchLayer(Module):
    def __init__(self,qnode,weights):
        if not TORCH_IMPORTED:
            raise ImportError(
                "TorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()

        #weight_shapes = {
        #    weight: (tuple(size) if isinstance(size, Iterable) else () if size == 1 else (size,))
        #    for weight, size in weight_shapes.items()
        #}

        # validate the QNode signature, and convert to a Torch QNode.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        #self._signature_validation(qnode, weight_shapes)
        self.qnode = qnode
        self.qnode.interface = "torch"

        self.qnode_weights = weights

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """

        if len(inputs.shape) > 1:
            # If the input size is not 1-dimensional, unstack the input along its first dimension,
            # recursively call the forward pass on each of the yielded tensors, and then stack the
            # outputs back into the correct shape
            reconstructor = [self.forward(x) for x in torch.unbind(inputs)]
            return torch.stack(reconstructor)

        # If the input is 1-dimensional, calculate the forward pass as usual
        return self._evaluate_qnode(inputs)


    def _evaluate_qnode(self, x):
        """Evaluates the QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        return torch.hstack(res).type(x.dtype)

    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__
    _input_arg = "inputs"

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg
    


class qrnn_lightning(torch.nn.Module):
    def __init__(self, anc_q, n_qub_enc, seq_num, D):
        """
        # n_qub_enc: # of of qubits for encodong input
        # anc_q: # of of qubits for memerizing history
        # seq_num: How many pieces the text is divided.
        # D: the # of layers of variational layers
        """
        super().__init__()
        self.num_anc_q=anc_q #RegH
        self.seq_num=seq_num #seq_len
        self.n_qub_enc=n_qub_enc #RegD
        self.num_ansatz_q=anc_q+n_qub_enc #RegH + RegD -> Ansatz
        self.num_para_per_bloc=self.num_ansatz_q*(3*D+2)
        #self.n_input_each_blc=n_qubs#2**n_qubs#(Denc+2)
        self.D=D
        #self.Denc=Denc
        self.num_q=self.n_qub_enc*self.seq_num+self.num_anc_q

        self.init_params=torch.nn.Parameter((np.pi/4)*(2*torch.randn(self.num_ansatz_q*(3*self.D+2)*self.seq_num) - 1))

        self.dev = qml.device("lightning.qubit", wires=self.num_q)
        self.qnod=qml.QNode(self.circuit, self.dev, interface="torch")
        self.weight = {"weights": self.init_params}
        self.linear = TorchLayer(self.qnod, self.weight)

    def circuit(self,inputs,weights):
        index=0
        for i in range(self.seq_num):
            start=i*self.n_qub_enc
            end=(i+1)*self.n_qub_enc
            self.encoding(inputs[start:end])
            self.ansatz(weights[i*self.num_para_per_bloc:(i+1)*self.num_para_per_bloc])
            index+=self.num_ansatz_q*(3*self.D+2)*self.seq_num
            if i!=self.seq_num-1:
                for j in range(self.n_qub_enc):
                    q1=j+self.num_anc_q
                    q2=(i+1)*self.n_qub_enc+j+self.num_anc_q
                    #print(q1,q2)
                    qml.SWAP(wires=[q1,q2])
                    #m1=qml.measure(q2)

        return qml.expval(qml.PauliZ(0))

    def ansatz(self,weights):
        indx=0
        for j in range(self.num_ansatz_q):
            qml.RX(weights[indx],wires=j)
            qml.RZ(weights[indx+1],wires=j)
            qml.RX(weights[indx+2],wires=j)
            indx+=3
        for i in range(self.D):
            for j in range(self.num_ansatz_q):
                qml.IsingZZ(weights[indx],wires=[j,(j+1)%self.num_ansatz_q])
                indx+=1
            for j in range(self.num_ansatz_q):
                qml.RY(weights[indx],wires=j)
                indx+=1

    def encoding(self, input):
        indx=0
        for j in range(self.n_qub_enc):
            qml.RY(input[indx],j+self.num_anc_q)
            indx+=1


    def forward(self, x):

        x=self.linear(x)
        return torch.sigmoid(x)
    def retrive_init(self):
        return ''.join([str(self.num_anc_q), str(self.n_qub_enc), str(self.seq_num), str(self.D)])
        
    def draw_circuit(self):
        from pennylane import draw

        # Create a temporary QNode for drawing
        dev = qml.device("default.qubit", wires=self.num_q)

        @qml.qnode(dev)
        def circuit_to_draw(inputs, weights):
            return self.circuit(inputs, weights)

        # Dummy inputs with correct shapes
        dummy_input = torch.randn(self.n_qub_enc * self.seq_num)
        dummy_weights = torch.randn_like(self.init_params)

        # Draw and print
        drawer = draw(circuit_to_draw)
        print(drawer(dummy_input, dummy_weights))