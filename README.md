This folder contains scripts and notebooks to process healthy control MC10 sensor data
for activity recognition classification.

`requirements.txt`
1. List of package and version requirments that need to be installed to run scripts.
1. To install dependencies run: `pip install -r requirements.txt`
1. To create new dependencies file run in correct directory: `pip freeze > requirements.txt`

`activityrec.ipynb`

``





Remove
```
PreprocessFcns.py
- clip generating function (gen_clips)
- 37 feature extraction function (feature_extraction)
- includes high, band, low pass filters

TestFeatures.ipynb
- input: activity dictionary
- use gen_clips
- use feature_extraction
- check 'features'

data structure of clipped data
- using nested dictionary: example[trial][sensor]
  - trial: 0, 1, etc
  - sensor: accel, gyro
  - data
  - clip_len
```