from tensorflow.keras.models import load_model
import numpy as np
from music_preprocessor import SEQUENCE_LENGTH, load, MAPPING_PATH
import json

# Constants
OUTPUT_UNITS = 38 
MAX_NOTES = 100  

# Load the trained model
model = load_model("model.h5")

# Load the mapping (from integer to symbol)
with open(MAPPING_PATH, 'r') as f:
    mapping = json.load(f)

# Reverse mapping (from symbol to integer)
reverse_mapping = {v: k for k, v in mapping.items()}

def generate_music(model, seed_sequence, max_notes=MAX_NOTES):
    generated_sequence = []
    
    current_sequence = seed_sequence
    
    for _ in range(max_notes):
        
        prediction = model.predict(current_sequence)
        
        predicted_index = np.argmax(prediction, axis=-1)[0]
        
        generated_sequence.append(predicted_index)
        
        predicted_note = keras.utils.to_categorical(predicted_index, num_classes=len(mapping))
        predicted_note = predicted_note[np.newaxis, np.newaxis, ...]
        current_sequence = np.concatenate([current_sequence[:, 1:, :], predicted_note], axis=1)
    
    return generated_sequence

if __name__ == "__main__":
    seed_sequence = [64, 64, 65, 67, 69, 67, 65, 64]  # C4, C4, D4, E4, G4, E4, D4, C4

    
    seed_sequence = keras.utils.to_categorical(seed_sequence, num_classes=len(mapping))
    
    # Reshape to match the model input
    seed_sequence = seed_sequence[np.newaxis, ...]  # Add batch dimension

    # Generate music
    generated_sequence = generate_music(model, seed_sequence)
    
    # Convert the generated sequence from integers to symbols
    generated_music = ' '.join(map(str, [reverse_mapping[i] for i in generated_sequence]))
    
    print("Generated Music Sequence:", generated_music)
