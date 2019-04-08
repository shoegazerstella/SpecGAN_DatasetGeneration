# SpecGAN_DatasetGeneration
Dataset generation for training a SpecGAN

This script takes a folder full of audio files and extracts harmonic and percussive samples, creating a dataset useful for training a SpecGAN based on your favourite music.

I've used [Librosa](https://librosa.github.io/librosa/index.html) Harmonic/Percussive separation, then I adapted [this](https://gist.github.com/kylemcdonald/c8e62ef8cb9515d64df4) script for samples separation based on onset detection.

I then used [this](https://github.com/naotokui/SpecGAN) repo for training my SpecGAN.

### Usage
```
python genDataset.py -i myAudioDir/ -o myOutputDir/
```

#### Samples
Here the [samples dir](/samples/) with some percussive examples extracted from [this song](https://www.youtube.com/watch?v=ybzSWlpgJOA).