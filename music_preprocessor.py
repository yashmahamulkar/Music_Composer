import os
from music21 import converter, note, key
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

FOLDER_PATH = 'erk'
SAVE_DIR = 'dataset'
SINGLE_FILE_DATASET = 'Master_dataset'
MAPPING_PATH = 'mapped_songs.json'
SEQUENCE_LENGTH = 64

# Loading Songs from Folder
def load_songs(folder_path):
    songs = []
    for root, dirs, files in os.walk(folder_path):
        files.sort()
        for file in files:
            if file.endswith('.krn'):
                file_path = os.path.join(root, file)
                song = converter.parse(file_path)
                songs.append(song)
    return songs

# Transposing the song like (EG: if song is in BMaj the we convert it to CMaj)
def transpose_song(song):
  song_keys = []
  song_sharpness = []

  key_sig = song.flat.getElementsByClass(key.Key)
  key_signatures = song.flat.getElementsByClass(key.KeySignature)

  parts = song.getElementsByClass(m21.stream.Part)
  measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
  main_key = measures_part0[0][4]

  if not isinstance(main_key, m21.key.Key):
      main_key = song.analyze('key')

  if main_key.mode == 'major':
      interval = m21.interval.Interval(main_key.tonic, m21.pitch.Pitch('C'))

  elif main_key.mode == 'minor':
      interval = m21.interval.Interval(main_key.tonic, m21.pitch.Pitch('A'))

  for ks in key_sig:
    key_name = ks.name
    song_keys.append(key_name)
  for ks in key_signatures:
      key_sharphess = ks.sharps
      song_sharpness.append(key_sharphess)

  transpose = song.transpose(interval)

  return transpose, song_keys, song_sharpness


# Extracting the pitches and Duration from the song
def encode_song(song, time_step = 0.25):
    encoded_song = []

    for element in song.flat.notesAndRests:
            if isinstance(element, m21.note.Note):
                symbol = element.pitch.midi

            elif isinstance(element, m21.note.Rest):
                symbol = 'r'

            steps = int(element.duration.quarterLength / time_step)
            for step in range(steps):
                if step == 0:
                    encoded_song.append(symbol)
                else:
                    encoded_song.append('_')

    encoded_song = ' '.join(map(str, encoded_song))
    return encoded_song

def load(file_path):
  with open(file_path, 'r') as f:
    song = f.read()
  return song

# saving all the songs in a single dataset_file
def create_single_dataset_file(dataset_path, file_dataset_path, sequence_length=64):
  new_song_delimiter = '/ ' * sequence_length
  songs = ""

  # load the encoded music
  for path, _, files in os.walk(dataset_path):
    for file in files:
      file_path = os.path.join(path, file)
      song = load(file_path)
      songs = songs + song + ' ' + new_song_delimiter + ' '
  songs = songs[:-1]

  # save string that contains all dataset
  with open(file_dataset_path, 'w') as f:
    f.write(songs)

  return songs


def create_mapping(songs, mapping_path):
  mapping = {}
  #identifying the vocabulary
  songs = songs.split()
  vocabulary = list(set(songs))
  for i, symbol in enumerate(vocabulary):
    mapping[symbol] = i

  #saving it in json file
  with open(mapping_path, 'w') as f:
    json.dump(mapping, f, indent=4)


# converted this song 64_ _ _ 60 to 0,0,1,12,...
def convert_songs_to_int(songs):
  int_songs = []

  #load the mapping
  with open(MAPPING_PATH, 'r') as f:
    mapping = json.load(f)

  # cast songs string to list
  songs = songs.split()

  # map songs to int
  for symbol in songs:
    int_songs.append(mapping[symbol])

  return int_songs


# spiltting data into training sequence
def generating_training_sequences(sequence_length):
  songs = load(SINGLE_FILE_DATASET)
  int_songs = convert_songs_to_int(songs)

  inputs = []
  targets = []

  # generate train split
  num_sequences = len(int_songs) - sequence_length
  for i in range(num_sequences):
    inputs.append(int_songs[i:i+sequence_length])
    targets.append(int_songs[i+sequence_length])

  # encoding the sequences
  vocabulary_size = len(set(int_songs))
  inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
  targets = np.array(targets)

  return inputs, targets

"""Preprocessing Function everything preprocesses here."""

def pre_prosessor(filepath):

  # Loading all the songs from the folder
  songs = load_songs(filepath)

  # Have loaded all the songs and printed its length
  print('number of songs', len(songs))


  # Preprocessing of the songs happens here
  final_song_keys = []
  final_song_sharpness = []


  for i, song in enumerate(songs):

    #transposing the song in this form: [60 _ _ _ 30 _ _]
    song, song_keys, song_sharpness = transpose_song(song)

    # Extracting the keys and sharpness from the song
    final_song_keys.append(song_keys)
    final_song_sharpness.append(song_sharpness)

    # Encoding the song
    encoded_song = encode_song(song)

    # print(encoded_song)

    # Saving the song in a file
    save_path = os.path.join(SAVE_DIR, str(i))
    with open(save_path, 'w') as f:
      f.write(encoded_song)

  # print("keys name:", final_song_keys)
  # print("keys Sharpness:", final_song_sharpness)

  # transposed_song.show() Works only in your desktop environment

"""main function of the whole code"""

if __name__ == '__main__':
  pre_prosessor(FOLDER_PATH)

  songs = create_single_dataset_file(SAVE_DIR, SINGLE_FILE_DATASET)

  create_mapping(songs, MAPPING_PATH)

  # inputs, targets = generating_training_sequences(SEQUENCE_LENGTH)

  # print(f"inputs: {inputs.shape}")
  # print(f"Splitted_target: {targets.shape}")