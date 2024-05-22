from decode_files import FileDecoder
from extract_features import StatFeatExtractor

class AudioFile:
  def __init__(self, file_path):
    self.path = file_path
    self.decoder, self.extractor = FileDecoder(file_path), StatFeatExtractor(file_path)
    self._x_info = 0
    self._y_info = 0
    self._vehicle_type = 0
    self.is_background = self.decoder.is_background()

  def y_info(self):
    if self._y_info == 0:
      self._y_info = self.decoder.decode_filename()
    return self._y_info
  
  def x_info(self):
    if self._x_info == 0:
      self._x_info = self.extractor.extract_feats()
    return self._x_info

  def vehicle_type(self):
    if self._vehicle_type == 0:
      self._vehicle_type = self.decoder.decode_filename()['vehicle']
    return self._vehicle_type
