"""
### Author: Jacob Parmer
###
### Created: Mar 20, 2021
"""
import torch
from torch import nn

from src.time_logs import TimerLog

class RNN(nn.Module):
    """
    Implements a Recurrent Neural Network model.

    Largely based on Nicklas Hansen's model described in 'Voie Activity Detection in Noisy Environments'.
    Utilizes LSTM's and CrossEntropyLoss.
    """

    def __init__(self, input_size, hidden_size, num_layers, device, verbose=False):
        """
        Creates the RNN using LSTM.

        LeakyReLU() was used as activation function because ReLU() was defaulting values to 0 after some training.
        This may or may not actually be useful. Probably should be tested with Tanh() and ReLU() and compared at some point. 

        Params:
            (int) input_size: Size of neural network inputs - should be equal to the frame size being passed in
            (int) hidden_size: Size of neural network hidden layers - larger sizes means longer run time
            (int) num_layers: Number of neural network layers, basically stacking RNN's on top of each other
            (str) device: Allows network to use cuda if available, 'cuda' or 'cpu'
            (bool) verbose: Displays running information to CLI while running if this is enabled
        """
        
        super(RNN, self).__init__()

        self.input_size = input_size
        self.device = device
        self.to(self.device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.verbose = verbose

        if self.verbose:
            print(f'Using {self.device} device')
        
        self.relu = nn.LeakyReLU()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.lin1 = nn.Linear(hidden_size, 26)
        self.lin2 = nn.Linear(26, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Performs the forward pass of the network

        Params:
            (tensor) x: Input data, should be 3D tensor of shape [# frames, batch_size, frame size]

        Returns:
            (tensor) self.softmax(x): Neuron outputs
        """

        self.hidden = self.init_hidden(x.size(1))
        x, _ = self.rnn(x, self.hidden)
        
        x = x.contiguous().view(x.size(0), -1)

        x = self.relu(self.lin1(x))
        x = self.lin2(x)

        return self.softmax(x)

    def init_hidden(self, batch_size):
        """
        Initializes the hidden layer(s) of the network

        Params:
            (int) batch_size: Size of batch - in an unedited version of this implementation this will always be 1.

        Returns:
            (tensor) h, c: zero tensors of size [num_layers, batch_size, hidden_size]
        """

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h, c

    def train(self, librispeech, epochs=1, lrate=0.01, verbose=False):
        """
        Trains the network on provided data

        Haven't yet seen a great improvement in performance with higher epochs, which is why it's defaulted to 1.
        Also because higher epochs take an eternity to train.

        Params:
            (LIBRISPEECH) librispeech: Contains audio data information as well as other useless things.
            (int) epochs: Number of times to run training on given dataset. Increasing this drastically lengthens required training time.
            (float) lrate: Learning rate for Adam optimizer
            (bool) verbose: Displays running information to CLI while running if this is enabled
        """

        timer = TimerLog()

        if verbose:
            print(f"Starting training for {epochs} epochs...")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        
        for epoch in range(1, epochs+1):
            for i in range(len(librispeech.dataset)):
                X, y = librispeech.load_data(i, n_mels=self.input_size, n_mfcc=self.input_size)
                optimizer.zero_grad()
                output = self(X)

                loss = criterion(output, y.view(-1).long())
                loss.backward()
                optimizer.step()

                if verbose:
                    print(f"#{i}/{len(librispeech.dataset)}")
                    print(f"Loss: {loss.item():.5}")
                    if i % len(X) == 0:
                        print(f"Sample Output:\n{torch.argmax(output, dim=1)}")
                        print(f"Targets:\n{y}")

            if verbose:
                print(f"epoch: {epoch}/{epochs}")

        if verbose:
            print(f"Finished model training for {librispeech.name} in {timer.get_elapsed()} seconds.")

        return

    def test(self, librispeech, verbose=False):
        """
        Tests the network on provided data.

        Displays output for accuracy, FAR, and FRR even if verbose is turned off.
        FAR, False Rejection Rate, is the % of the time 0 is predicted when the output is supposed to be 1.
        FRR, False Acceptance Rate, is the % of the time 1 is predicted when the output is supposed to be 0.

        Params:
            (LIBRISPEECH) librispeech: Contains audio data information as well as other useless things.
            (bool) verbose: Displays running information to CLI while running if this is enabled
        """
        total_classifications = 0
        accuracy = 0
        FRR = 0
        FAR = 0

        for i in range(len(librispeech.dataset)):
            X, y = librispeech.load_data(i, n_mels=self.input_size, n_mfcc=self.input_size)
            y = y.view(-1)

            output = self(X)

            for j, frame in enumerate(output):
                total_classifications += 1
                prediction = torch.argmax(frame)

                if prediction == y[j]:
                    accuracy += 1
                elif prediction == 0 and y[j] == 1:
                    FRR += 1
                elif prediction == 1 and y[j] == 0:
                    FAR += 1

            if verbose:
                print(f"#{i}/{len(librispeech.dataset)}")
                print(f"Correct classifications: {accuracy}")
                print(f"Total classifications: {total_classifications}")

        accuracy = accuracy / total_classifications
        FRR = FRR / total_classifications
        FAR = FAR / total_classifications

        print(f"Accuracy over test dataset {librispeech.name}: {accuracy*100:.2f}%")
        print(f"False Rejection Rate (FRR): {FRR*100:.2f}%")
        print(f"False Acceptance Rate (FAR): {FAR*100:.2f}%")