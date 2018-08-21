# activity-recognition-process


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
