# Backpropamine: differentiable neuromdulated plasticity.
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License file in this repository for the specific language governing
# permissions and limitations under the License.

# The Network class implements a "backpropamine" network, that is, a neural
# network with neuromodulated Hebbian plastic connections that is trained by
# gradient descent. The Backpropamine machinery is
# entirely contained in the Network class (~25 lines of code).

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class BackpropamineRNN(nn.Module):
    # RNN with trainable modulated plasticity ("backpropamine")

    def __init__(self, isize, hsize, osize, freeze_plasticity=False):
        super(BackpropamineRNN, self).__init__()
        self.hsize, self.isize = hsize, isize

        self.freeze_plasticity = freeze_plasticity

        self.i2h = torch.nn.Linear(
            isize, hsize
        )  # Weights from input to recurrent layer
        self.w = torch.nn.Parameter(
            0.001 * torch.rand(hsize, hsize)
        )  # Baseline (non-plastic) component of the plastic recurrent layer

        self.alpha = torch.nn.Parameter(
            0.001 * torch.rand(hsize, hsize)
        )  # Plasticity coefficients of the plastic recurrent layer; one alpha coefficient per recurrent connection
        # self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))  # Per-neuron alpha
        # self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))         # Single alpha for whole network

        self.h2mod = torch.nn.Linear(
            hsize, 1
        )  # Weights from the recurrent layer to the (single) neurodulator output
        self.modfanout = torch.nn.Linear(
            1, hsize
        )  # The modulator output is passed through a different 'weight' for each neuron (it 'fans out' over neurons)

        self.h2o = torch.nn.Linear(
            hsize, osize
        )  # From recurrent to outputs (e.g. action probabilities in RL)

        ## value prediction (for A2C)
        # self.h2v = torch.nn.Linear(hsize, 1)            # From recurrent to value-prediction (used for A2C)

    def forward(self, inputs, hidden_hebbian):
        # hidden[0] is the h-state (recurrent); hidden[1] is the Hebbian trace
        h_cur, hebb = hidden_hebbian

        # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
        h_next = F.tanh(
            self.i2h(inputs)
            + h_cur.unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)
        )  # Update the h-state
        outputs = self.h2o(h_next)  # pure linear output

        ## value prediction (for A2C)
        # valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...
        deltahebb = torch.bmm(
            h_cur.unsqueeze(2), h_next.unsqueeze(1)
        )  # Batched outer product of previous hidden state with new hidden state

        # We also need to compute the eta (the plasticity rate), wich is determined by neuromodulation
        # Note that this is "simple" neuromodulation.
        myeta = F.tanh(self.h2mod(h_next)).unsqueeze(2)  # Shape: BatchSize x 1 x 1

        # The neuromodulated eta is passed through a vector of fanout weights, one per neuron.
        # Each *column* in w, hebb and alpha constitutes the inputs to a single cell.
        # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
        # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each
        # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
        # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
        myeta = self.modfanout(myeta)

        # Updating Hebbian traces, with a hard clip (other choices are possible)
        self.clipval = 2.0
        hebb_next = torch.clamp(
            hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval
        )

        return outputs, (h_next, hebb_next)

    def initialZeroStateHebb(self, batch_size):
        return self.initialZeroState(batch_size), self.initialZeroHebb(batch_size)

    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False)

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(
            torch.zeros(BATCHSIZE, self.hsize, self.hsize), requires_grad=False
        )
