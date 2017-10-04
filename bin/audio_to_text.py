import json
import os
import sys
import subprocess
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description = 'Converts audio segments to text using DeepSpeech')
    parser.add_argument('--path_to_model_graph','-model_path',
                        help = 'Set the path to the inferring model\'s .pb graph.')
    parser.add_argument('--path_to_audio_folder','-audio_path',
                        help = 'Set path to folder containind .wav audio segments.')
    parser.add_argument('--alphabet_path', '-abc',
                        help = 'Set path to file containind model\'s alphabet.')
    parser.add_argument('--path_to_transcript','-text_path',
                        help = 'Set the path for the output transcript file')

    inputs = vars(parser.parse_args())

    model_path = inputs['path_to_model_graph']
    audio_path = inputs['path_to_audio_folder']
    abc_path = inputs['alphabet_path']

    # validate model path
    if not os.path.exists(model_path.encode('utf-8')):
        print 'No model found at given location: ' + model_path
        return
    else:
        model_path = os.path.abspath(model_path.encode('utf-8'))

    # validate abc path
    if not os.path.exists(abc_path.encode('utf-8')):
        print 'No alphabet file found at given location: ' + abc_path
        return
    else:
        abc_path = os.path.abspath(abc_path.encode('utf-8'))
    
    # get audio files
    audio_segment_paths = []
    audio_segment_names = []

    for maybe_audio in os.listdir(audio_path.encode('utf-8')):
        if maybe_audio.endswith('.wav'):
            audio_segment_name = maybe_audio
            audio_segment_names.append(audio_segment_name)
            rel_segment_path = os.path.join(audio_path,maybe_audio)
            abs_segment_path = os.path.abspath(rel_segment_path)
            audio_segment_paths.append(abs_segment_path)

    # call the demo native client for each given audio segment and catch stout = transcription
    inferrence_records = []

    DEEPSPEECH_DEMO_PATH = '/home/ubuntu/setup/deepspeech'

    print 'Starting inference ...'

    # add path 
    os.system('LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH')

    # start inference
    for audio_segment_name,audio_segment_path in zip(audio_segment_names, audio_segment_paths):
        command = [DEEPSPEECH_DEMO_PATH, model_path, audio_segment_path, abc_path]
        deepspeech = subprocess.Popen(command, stdout = subprocess.PIPE)
        print 'Subprocess started. Accessing its output...'
        inferred_transcript = deepspeech.communicate()[0].encode('utf-8').strip()
        print 'stdout from subprocess: ' + inferred_transcript.strip()
        segment_record = {'audio_path':audio_segment_path,
                          'audio_name':audio_segment_name,
                          'audio_transcript': inferred_transcript}
        inferrence_records.append(segment_record)

    output_name = 'inferrence_records.json'

    print 'Inference finished. Saving transcripts as ' + output_name + ' ...'

    with open(output_name,'w') as file:
        json.dump(inferrence_records,file)

    print 'Transcipts saved as ' + output_name + '. Done.'

if __name__ == '__main__':
    main()
