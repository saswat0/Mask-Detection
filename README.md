# COVID-19 Face Mask Detector

# Reproduction
```Shell
git clone https://github.com/saswat0/Mask-Detection.git
cd Mash-Detection
```

**For testing on PyTorch Model**
```bash
python test.py
```

# Download dataset and export it to pandas DataFrame
```
python data_preparation
```
## Training

```Shell
python train
```

## Testing on videos
```sh
python video modelPath videoPath
```

### Usage
```
Usage: video.py [OPTIONS] MODELPATH VIDEOPATH

  modelPath: path to model.ckpt

  videoPath: path to video file to annotate. Index 0 to access webcam

Options:
  --output PATH  specify output path to save video with annotations
```