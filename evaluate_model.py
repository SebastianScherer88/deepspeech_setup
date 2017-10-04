from util.text import wers
from util.text import meds
from argparse import ArgumentParser
import pandas as pd
import os
import subprocess
import numpy as np

def infer_model(model_path, audio_paths, transcripts, abc_path):

    inferred_transcripts = []
    n_samples = len(model_path)
        # start inference
    for i,(audio_path, transcript) in enumerate(zip(audio_paths,transcripts)):
        print 'audio path: ' + str(audio_path)
        print 'transcript: ' + str(transcript)
        print 'abc path: ' + str(abc_path)
        #command = [DEEPSPEECH_PATH, model_path, audio_path, abc_path]
        global DEEPSPEECH_PATH
        command = DEEPSPEECH_PATH+'deepspeech '+model_path+' '+str(audio_path)+' '+abc_path
        print 'command: ' + str(command)
        print '======================================='
        deepspeech = subprocess.Popen(command, stdout = subprocess.PIPE,shell=True)
        print 'Inferring audio '+str(i)+'/'+str(n_samples)
        inferred_transcript = deepspeech.communicate()[0].encode('utf-8').strip()
        print 'Inferred transcript: ' + inferred_transcript
        print 'Actual transcript: ' + str(transcript)
        print '-------------------------------------------------'
        inferred_transcripts.append(inferred_transcript)

    return inferred_transcripts

def main():
    parser = ArgumentParser(description = 'Evaluate a deepspeech model on a given dataset.')
    parser.add_argument('--model_path','-mp',
                        help = 'Set the absolute path to the model graph.pb file.')
    parser.add_argument('--data_path','-dp',
                        help = 'Set the absolute path to the .csv file containind ground truth transcripts and audio paths in standardised format.')
    parser.add_argument('--abc_path','-ap',
                        help = 'Set the absolute path to the alphabet.txt file used during model training.')
    parser.add_argument('--out_dir','-out',
                        help = 'Set the directory for the output csv.',
                        default = './')
    # maybe introduce WER and MED flags for more flexibility later
    args = vars(parser.parse_args())

    model_path = args['model_path']
    # get model name of specified model
    model_name = model_path.split('/')[-1]

    data_path = args['data_path']
    # get name of specified csv file
    data_name = data_path.split('/')[-1]

    if not data_name.endswith('.csv'):
        print 'File must be of ".csv" type!'
        return
    abc_path = args['abc_path']
    output_dir = args['out_dir']

    # validate save dir
    if not os.path.isdir(output_dir):
        print "The specified output directory does not exist!"
        return

    # get paths to audio files and the corresponding ground truth transcriptions from csv
    data_csv = pd.read_csv(data_path).to_dict(orient='list')

    print data_csv

    n_samples = len(data_csv['wav_filename'])

    # update python paths
    global DEEPSPEECH_PATH
    UPDATE_LIBRARY_PATH_1 ="LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH"
    DEEPSPEECH_PATH ="/home/ubuntu/setup/"
    UPDATE_LIBRARY_PATH_2 ="LD_LIBRARY_PATH="+DEEPSPEECH_PATH+":$LD_LIBRARY_PATH"
    os.system(UPDATE_LIBRARY_PATH_1)
    os.system(UPDATE_LIBRARY_PATH_2)

    # start inference loop
    true_transcripts = data_csv['transcript']
    data_csv['inferred_transcripts'] = infer_model(model_path, data_csv['wav_filename'], true_transcripts, abc_path)

    # apply performance metrics to (y_true, y_inf) pairs of strings
    wer_results, wer_average = wers(true_transcripts, data_csv['inferred_transcripts'])
    med_results, med_average = meds(true_transcripts, data_csv['inferred_transcripts'])

    data_csv['WER'] = wer_results
    data_csv['MED'] = med_results

    print 'Average WER: ' + str(wer_average)
    print 'Average MED: ' + str(med_average)

    data_frame = pd.DataFrame.from_dict(data_csv)
    out_path = output_dir+'/'+model_name+'_evaluated_'+data_name
    data_frame.to_csv(out_path,index=False)

if __name__ == '__main__':
    main()
