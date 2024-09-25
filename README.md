# 🎵 Music Composer using LSTM

Website - 

## Description

This project aims to generate music compositions using a Long Short-Term Memory (LSTM) neural network. The model is trained on sequences of musical notes, learning the patterns and relationships between them. After training, the model generates new sequences of notes, effectively composing unique pieces of music.
Our LSTM model is designed to predict the next note in a sequence, given a set of previous notes. This allows the model to generate a continuous stream of music. The generated music can then be converted into a MIDI file, allowing it to be played by any MIDI-enabled device or software.

---

## Features

- **Data Preprocessing**: Converts raw musical data into a format suitable for training the LSTM model.
- **LSTM Model**: Uses LSTM layers to learn and predict musical sequences.
- **Music Generation**: Generates new musical compositions based on the learned patterns.
- **MIDI Conversion**: Converts predicted notes back into a playable MIDI file format.
  
---

## Demo

### Images

![Model Architecture](path_to_image)
*Image 1: LSTM Model Architecture*

![Music Notes](path_to_image)
*Image 2: Sample generated music notes*

### Video

Click the image below to see a demo of the project in action!

[![Watch the video](path_to_thumbnail)](link_to_video)

---

## Model Precessions

| Trained on Epoch | Model Accuracy | Model Loss |
|------------------|----------------|------------|
| 50               | 92%            | 8%         |

---

## Installation and Usage

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- Music21 (for MIDI data processing)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/music_composer.git
   cd music_composer

2. setup the requirements.txt
   ```bash
   pip install -r requirements.txt

3. Run the main.py as follows:
   ```bash
   python main.py

## Thankfull Team Mebers
- [Aditya Patil](https://github.com/Dracgamer5643)
- [Sakshi More]()
- [Soham Khot]()
