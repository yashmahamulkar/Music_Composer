import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from music_preprocessor import SEQUENCE_LENGTH, MAPPING_PATH
import json
import numpy as np
from midiutil import MIDIFile
from midi2audio import FluidSynth
from pydub import AudioSegment

class music_generator:
    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mapping = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
    
    def generate_music(self, seed, num_steps, max_sequence_length, temperature):
        # creating seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        
        # mapping seeds to integers
        seed = [self._mapping[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            # limit the seed to max sequence length
            seed = seed[-max_sequence_length:]
            
            # encoding the seed
            encoded_seed = keras.utils.to_categorical(seed, num_classes=len(self._mapping))
            encoded_seed = encoded_seed[np.newaxis, ...]
            
            # making the predictions
            probabilities = self.model.predict(encoded_seed)[0]
            
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # update the seed
            seed.append(output_int)
            
            # map int to our coding
            output_symbol = [k for k, v in self._mapping.items() if v == output_int][0]
            
            # check if we are at the end of the melody
            if output_symbol == "/":
                break
                
            # update the melody
            melody.append(output_symbol)

        return melody
    
    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        
        return index

    def save_melody_to_midi(self, melody, file_name="melody.mid", tempo=220):
        midi = MIDIFile(1)  # create a single track MIDI file
        midi.addTempo(0, 0, tempo)

        # start time for the first note
        start_time = 0
        
        for note_str in melody:
            # Skip if it's a rest or unrecognized symbol
            if note_str.isdigit():  # Only process if the string is a digit
                note = int(note_str)
                midi.addNote(0, 0, note, start_time, 1, 100)  # channel, pitch, duration, volume
            start_time += 1

        # Write to the file
        with open(file_name, "wb") as output_file:
            midi.writeFile(output_file)

    def midi_to_mp3(self, midi_file, mp3_file):
        # Convert MIDI to WAV using FluidSynth
        fs = FluidSynth(r'fluidsynth-2.3.6-win10-x64\lib\FluidR3 GM.sf2')
        wav_file = midi_file.replace(".mid", ".wav")
        fs.midi_to_audio(midi_file, wav_file)

        # Convert WAV to MP3 using pydub
        sound = AudioSegment.from_wav(wav_file)
        sound.export(mp3_file, format="mp3")

if __name__ == '__main__':
    md = music_generator()
    seed = "62 _ 60 _ 60 _ 62 _"
    melody = md.generate_music(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    
    # Save the generated melody to a MIDI file
    midi_file = "generated_melody.mid"
    mp3_file = "generated_melody.mp3"
    md.save_melody_to_midi(melody, midi_file)
    
    # Convert the MIDI file to MP3
    md.midi_to_mp3(midi_file, mp3_file)
