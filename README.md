# VAD

This is an implementation of a Vocal Activity Detector using PyTorch.
In this, a Recurrent Neural Network (RNN) classifies frames of audio data into speech (1) or not-speech (0).

These are commonly used as the first stage of a pipeline for programs handling speech.
For example, if trying to do speech-recognition, it might be useful to just drop any data that does not contain anyone speaking, so that the model attempting to recognize speech isn't trying to figure out what the car in the background is saying.

## Installation

#### TL;DR

> git clone https://github.com/jacob-parmer/VAD.git
> 
> cd VAD
> 
> pip install -r requirements.txt

### Files
To create the working directory:

> git clone https://github.com/jacob-parmer/VAD.git

If you'd like, you can go ahead and install LibriSpeech/ files directly by downloading them from https://www.openslr.org/12 and placing the extracted folders in the LibriSpeech/ directory. This is not required since the program should automatically download these on the first run.

### Requirements
As with any machine learning project, the better specs your machine has, particularly in your CPU or GPU, the faster you're going to get results.
This project is no different, so while I don't think there are any actual requirements to have a strong machine here, I'd say it's highly recommended.

Aside from that, this program also requires a good bit of storage space to store all the LibriSpeech data. The data comes out to ~60Gb, and the .tar files aren't automatically deleted (yet), so it ends up being double this. (~120 Gb!) If you want to reduce this space,

A. Get rid of the .tar files. They aren't used after they've been extracted. 

B. in data.py, the datasets used are set in the build_librispeech() function. You can change these to use smaller datasets if needed.

All needed libraries should be available for download through the requirements.txt file. To get these, run the following command in the root directory:

> pip install -r requirements.txt

## Running the program

The program is currently run directly through python, so running a command like the following in the root directory:

> python3 main.py -m training -tv -d cpu

This will begin the model training process. To test instead, use 'testing' in place of 'training' in the above command. 

A list of all possible command line arguments can be found by adding a -h after main.py

## Results

The following is a sample audio signal with overlapping labels. Where labels are high, the prediction was speeched, and vice-versa.
![waveform](/images/labeled-audio.png)

It's definitely not flawless, but it does an alright job. I think it could improve with some more in-depth training, and possibly some pre-processing of the signal on the front-end. 

For a more complete view of the success of the model, the overall accuracy, false rejection rate, and false acceptance rate, were found using both clean and noisy test datasets provided by LibriSpeech. As one might expect, the model struggled more with noisy data, but not to a super significant degree.

![Testing](/images/results.png)

## To-do list / Known Issues

[ ] Make project work with CUDA 

[ ] Windows compatibility

[ ] Try runs with different activation functions? LeakyReLU vs. ReLU vs. Tanh

[ ] Dockerize / Makefile the application

[ ] Better handle feature / model shapes. Some of these are weird.

[ ] Make timed print work with breakpoints


## References
Here are some resources that I commonly referenced when working on this:

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://github.com/nicklashansen/voice-activity-detection

https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
