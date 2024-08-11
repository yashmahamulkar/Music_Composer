from flask import Flask, request, jsonify,render_template
from gen2 import music_generator
app = Flask(__name__)

@app.route('/send_notes', methods=['POST'])
def receive_notes():
    note_array = request.json.get('notes', [])
    temp = request.json.get('temperature')
    time= request.json.get('time')
    print(f"Received notes: {note_array}")
    notestring=""
    for i in note_array:
        notestring+=i.strip()+" _ "
    # Here you can process the note_array as needed
    print(notestring)
    md = music_generator()
    
    melody = md.generate_music(notestring, int(time), 64, float(temp))
    print(melody)
    
    # Save the generated melody to a MIDI file
    midi_file = "generated_melody.mid"
    mp3_file = "generated_melody.mp3"
    md.save_melody_to_midi(melody, midi_file)
    
    # Convert the MIDI file to MP3
    md.midi_to_mp3(midi_file, mp3_file)

    return f"Notes received: {note_array}"

@app.route('/')
def index():
    return render_template('/index.html')

if __name__ == '__main__':
    app.run(debug=True)
