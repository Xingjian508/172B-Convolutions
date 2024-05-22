class FileDecoder:
  def __init__(self, file_path):
    self.path = file_path
    self.info = self.pre_decode_filename()

  def decode_filename(self):
    return self.info
  
  def is_background(self):
    return self.info['is_background']

  def extract_y_features(self):
    """Extracts known features embedded within file name of dataset instance
  
    Args:
    path (str): stringified features
      example: "2019-10-22-08-40_Fraunhofer-IDMT_30Kmh_1116695_M_D_CR_ME_CH12.wav"
  
    Returns:
    array[str]: itemized features
      example: ['2019-10-22-08-40', 'Fraunhofer-IDMT', '30', '1116695', 'M', 'D', 'C', 'R', 'ME', '12']
    """
  
    features = self.path[:-4].split('_')
    features[2] = features[2][:-3]
    features[-1] = features[-1][2:]
    features = features[:6] + [features[6][0],features[6][1]] + features[7:]
    return features
  
  def extract_y_features_bg(self):
    features = self.path[:-7].split('_')
    return features
  
  def pre_decode_filename(self):
    is_background = self.path.endswith("-BG.wav")
    
    if is_background:
      features = self.extract_y_features_bg()
      info = {
        'date_time': features[0],
        'location': features[1],
        'speed': features[2],
        'sample_position': features[3],
        'is_background': is_background,
        'microphone_type' : features[4],
        'channels': features[5]
      }
      # print(info)
    else:
      features = self.extract_y_features()
      # print(f'Features: {features}')
      info = {
        'date_time': features[0],
        'location': features[1],
        'speed': features[2],
        'sample_position': features[3],
        'is_background': is_background,
        'daytime': features[4],
        'weather': features[5],
        'vehicle': features[6],
        'direction': features[7],
        'microphone_type': features[8],
        'channels': features[9] 
      }
    
    return info
