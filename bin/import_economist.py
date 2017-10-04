#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:43:06 2017

@author: sebastian
"""

import shutil
import tarfile
import sys
import os
import inflect
import copy
import wave
import audioop
from sox import Transformer
from pydub import AudioSegment

#sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas

print 'Current working directory:' + os.getcwd()

# pull back to DeepSpeech_new
#sys.path.insert(1, os.path.join(sys.path[0], '..'))

from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from pydub import AudioSegment
import json
from os import makedirs, path
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from nltk.tokenize import sent_tokenize

N_OK_AUDIOS = 0
N_CRIT_SHORT_AUDIOS = 0
N_CRIT_LONG_AUDIOS = 0
N_CRIT_POS_AUDIOS = 0
N_CRIT_MISMATCH = 0
N_NO_TRANSCRIPT = 0

def _preprocess_data(year, min_dur_secs, max_dur_secs, mode = 'remote'):

    # validate input
    if year not in ['2013','2014','2015','2016','2017'] or min_dur_secs >= max_dur_secs:
        print '''Input has to be like so: "economist_parser year min_dur max_dur". Example:
               "python -u economist_parser 2013 2 25"'''
        return

    # STEP 0: Set and create dir depending on server or local machine use
    data_dir = './data/economist/'+year

    if not gfile.Exists(data_dir):
        os.mkdir(data_dir)

    # STEP 1: Conditionally download data if mode == remote
    ZIP_FILE = 'economist_'+year+'_article_audio_matches.tar.gz' # or something
    if mode == 'remote':
        _maybe_download(data_dir, ZIP_FILE)

    # STEP 2: Conditionally extract (downloaded) zip file
    EXTRACTED_DIR = 'economist_'+year+'_article_audio_matches' # or something
    _maybe_extract(data_dir,
                   EXTRACTED_DIR,
                   ZIP_FILE)
    
    
    # STEP 3:Conditionally format articles into PLAIN format (needed for aeneas)
    ECONOMIST_MATCHES_ORIGINAL = 'article_audio_matches_'+year+'.json'
    ECONOMIST_MATCHES_PLAIN = 'article_audio_matches_'+year+'_plain.json'
    PLAIN_ARTICLES_DIR = 'plain_articles'
    print 'Maybe formatting articles'
    _format_articles(data_dir,
                           EXTRACTED_DIR,
                           PLAIN_ARTICLES_DIR,
                           ECONOMIST_MATCHES_ORIGINAL,
                           ECONOMIST_MATCHES_PLAIN)
    
    # STEP 4: Conditionally create syncmaps for all original audios (mp3 format)
    ECONOMIST_MATCHES_SYNC = 'article_audio_matches_'+year+'_sync.json'
    SYNC_MAPS_DIR = 'sync_maps'
    print 'Maybe making sync maps'
    _sync_articles(data_dir,
                          EXTRACTED_DIR,
                          SYNC_MAPS_DIR,
                          ECONOMIST_MATCHES_PLAIN,
                          ECONOMIST_MATCHES_SYNC)

    # if local, just create plain format text files and sync map text files to be included in the zip file that will be shelved; that will allow for faster data import on the server side later
    if mode == 'local':
        return
    
    # STEP 5: Split audios into sentence segments
    print 'Splitting audios'
    economist_files = split_audios(data_dir,
                                   EXTRACTED_DIR,
                                   ECONOMIST_MATCHES_SYNC,
                                   min_dur_secs,
                                   max_dur_secs)
    print 'Saving to csv'
    train_file_name = 'economist_'+year+'-train.csv'.encode('utf-8')
    economist_files.to_csv(path.join(data_dir.encode('utf-8'), train_file_name).encode('utf-8'), index=False)

    global N_CRIT_SHORT_AUDIOS
    global N_CRIT_LONG_AUDIOS
    global N_CRIT_POS_AUDIOS
    global N_NO_TRANSCRIPT
    global N_OK_AUDIOS
    print 'Segments ignored because too short: ' + str(N_CRIT_SHORT_AUDIOS)
    print 'Segments ignored because too long: ' + str(N_CRIT_LONG_AUDIOS)
    print 'Segments ignored because too early or late to guarantee alginment: ' + str(N_CRIT_POS_AUDIOS)
    print 'Segments ignored because of empty transcript: ' + str(N_NO_TRANSCRIPT)
    print 'Segments created with length > ', min_dur_secs, ' seconds: ' + str(N_OK_AUDIOS)
    

def _maybe_download(data_dir, zip_file):
    # if zip file not present in data_dir, download using gsutil
    if not gfile.Exists(path.join(data_dir, zip_file)):
        os.system('sudo gsutil cp gs://ais-data/speech2text/economist/'+zip_file+' '+data_dir)

def _maybe_extract(data_dir, extracted_dir, zip_file):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(path.join(data_dir, extracted_dir)):
        zip_path = path.join(data_dir, zip_file)
        tar = tarfile.open(zip_path)
        tar.extractall(data_dir)
        tar.close()
        
def _format_articles(data_dir, extracted_dir, plain_articles_dir, matches_orig, matches_plain):
    # If data_dir/extacted_data/plain_articles_dir does not exist, create it.
    # Then convert articles into aeneas' PLAIN format and dump there
    target_dir = path.join(data_dir,extracted_dir,plain_articles_dir)
    if gfile.Exists(target_dir):
        # if it exists, its the prepped=wrong one , i.e. delete and redo with correct preprocessing
        shutil.rmtree(target_dir, ignore_errors=True)
        # create target plain articles dir and updated json
        os.mkdir(target_dir)

    matches_orig_path = path.join(data_dir, extracted_dir, matches_orig)
    matches_plain_path = path.join(data_dir, extracted_dir, matches_plain)

    if gfile.Exists(matches_plain_path):
        os.remove(matches_plain_path)

    _make_plain_articles(matches_orig_path, matches_plain_path, target_dir)
        
def _make_plain_articles(articles_orig_path, articles_plain_path, save_dir):
    # Takes the match file.json located at matches_dir and creates aeneas PLAIN
    # versions of all articles, saving in save_dir. Updates articles.json
    print 'From within "make_plain_articles:"'
    with open(articles_orig_path, 'r') as matchfile:
        articles = json.load(matchfile)
    
    articles_editable = copy.deepcopy(articles)
    records = articles_editable['records']
    transformer = inflect.engine()

    for i,article in enumerate(articles['records']):
        print 'Iterating over articles: Article ' + str(i) + '\'s plain format is being created'
        text = article['text'].replace(u'\u2019',"'").replace(u'\u2014','-').replace(u'\u201c','"').replace(u'\u201d','"')
        plain_name = article['issue'].replace('-','_') + '_' + article['title'].lower().replace(' ','_') + '_' + article['fly_title'].lower().replace(' ','_') + '_plain_text'
        plain_name = plain_name.replace('/','_')
        plain_path = path.join(save_dir,plain_name)
        sentences = sent_tokenize(text)
        processed_sentences = [process_sentence(sentence, transformer) for sentence in sentences]
        
        with open(plain_path, 'w') as plain_file:
            for sentence in processed_sentences:
                plain_file.write(sentence.encode('utf-8'))
                plain_file.write('\n')
        # record location of plain article file just saved and a list of processed sentences
        articles_editable['records'][i]['text_plain_path'] = plain_path
        #articles_editable[i]['processed_sentences'] = processed_sentences
    #  save edited articles....json under new name with paths to plain format article content
    with open(articles_plain_path, 'w') as match_file_edited:
        json.dump(articles_editable, match_file_edited)
        
def _sync_articles(data_dir, extracted_dir, sync_maps_dir, matches_plain, matches_sync):
    # if data_dir/extracted_data/sync_maps does not exist create it.
    # The create sync maps with aeneas using original audios, plain format
    # articles and the articles.json to coordinate. and dump there
    target_dir = path.join(data_dir,extracted_dir,sync_maps_dir)
    if gfile.Exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
        os.mkdir(target_dir)

    matches_plain_path = path.join(data_dir, extracted_dir, matches_plain)
    matches_sync_path = path.join(data_dir, extracted_dir, matches_sync)

    if gfile.Exists(matches_sync_path):
        os.remove(matches_sync_path)

    _make_sync_maps(data_dir, extracted_dir, matches_plain_path, matches_sync_path, target_dir)
        
def _make_sync_maps(data_dir, extracted_dir, articles_plain_path, articles_sync_path, save_dir):
    # Takes the articles.json located at articles_pth and creates sync maps using
    # aeneas, saving in save_dir. Updates articles.json
    print 'From within "_make_sync_maps":'
    with open(articles_plain_path, 'r') as matchfile:
        articles = json.load(matchfile)
    
    config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
    
    articles_editable = copy.deepcopy(articles)
    
    for i, article in enumerate(articles['records']):
        # check if original audio's name has more that 15 characters, i.e. had a proper name and wont lead to name clashes -> missmatches
        orig_audio_name = article['path'].split('/')[-1]
        if len(orig_audio_name) <= 15:
            print 'This article points to an ambiguously named audio file. To prevent name clashes -> missmatches, this article will be skipped.'
            continue
        print 'Iterating over articles: Article ' + str(i) + '\'s sync map is being created'
        # since audio_name is path w.r.t. zip file's root dir, get audio path
        rel_audio_path = path.join(data_dir, extracted_dir, article['path'])
        audio_path = path.abspath(rel_audio_path)
        # get text path
        text_path = path.abspath(article['text_plain_path'])
        # get sync map save dir/name
        sync_name = article['issue'].replace('-','_') + '_' + article['title'].lower().replace(' ','_') + '_' + article['fly_title'].lower().replace(' ','_') + '_snyc_map'
        sync_name = sync_name.replace('/','_')
        smap_path = path.join(save_dir, sync_name)
        
        articles_editable['records'][i]['sync_map_path'] = smap_path

        task = Task(config_string=config_string)
    
        task.audio_file_path_absolute = audio_path
        task.text_file_path_absolute = text_path
        task.sync_map_file_path_absolute = smap_path

        #process task, create syncmap
        ExecuteTask(task).execute()

        # print created sync map to console
        task.output_sync_map_file()

    with open(articles_sync_path, 'w') as match_file_edited:
        json.dump(articles_editable, match_file_edited)
        
def split_audios(data_dir, extracted_dir, matches_sync, min_dur_secs, max_dur_secs):
    matches_sync_path = path.join(data_dir, extracted_dir, matches_sync)
    
    with open(matches_sync_path, 'r') as articles_file:
        articles = json.load(articles_file)
        
    files = []
    transformer = Transformer()

    for i, article in enumerate(articles['records']):
        if 'sync_map_path' not in article.keys():
            'This article does not have a sync map path (see "make_sync_maps" for details). Article skipped.'
            continue
        print 'Iterating over articles: Article ' + str(i) + '\'s audio segments are being created'
        print article['title']
        # get syncmap segments by parsing appropriate sync map file
        sync_map_path = article['sync_map_path']

        # skip notorious missmatches
        if article['path'].encode('utf-8') in ['2017-08-19/2017_08_19_after_charlottesville.mp3'.encode('utf-8'), #2017
                               '2017-05-20/2017_05_20_.mp3'.encode('utf-8'), #2017
                               '2017-05-06/2017_05_06_.mp3'.encode('utf-8'), #2017
                               '2016-08-20/2016_08_20_housing_in_america.mp3'.encode('utf-8'), #2016
                               '2016-07-30/2016_07_30_globalisation_and_politics.mp3'.encode('utf-8'), # 2016
                               '2016-07-09/2016_07_09_italian_banks.mp3'.encode('utf-8'), #2016
                               '2016-03-26/2016_03_26_business_in_america.mp3'.encode('utf-8'), #2016
                               '2016-01-09/2016_01_09_saudi_arabia.mp3'.encode('utf-8'), #2016
                               '2016-06-04/2016_06_04_free_speech.mp3'.encode('utf-8'), # 2015
                               '2015-03-28/2015_03_28_politics.mp3'.encode('utf-8'), #2015
                               '2015-02-21/2015_02_21_indias_economy.mp3'.encode('utf-8'), #2015
                               '2015-11-21/2015_11_21_paris.mp3'.encode('utf-8'), #2015
                               '2015-11-14/2015_11_14_the_world_economy.mp3'.encode('utf-8'), # 2015
                               '2015-06-27/2015_06_27_doctorassisted_dying.mp3'.encode('utf-8'), #2015
                               '2014-05-24/2014_05_24_narendra_modi.mp3'.encode('utf-8')]: # 2014
            continue

        sync_map_segments = parse_syncmap_file(sync_map_path, min_dur_secs, max_dur_secs)
        
        # get relative audio path, convert from mp3 to wav, and delete old mp3 version
        rel_audio_path_mp3 = path.join(data_dir, extracted_dir, article['path'])
        rel_audio_path_wav = '.' + ''.join(rel_audio_path_mp3.split('.')[:-1]) + '.wav'
        temp = AudioSegment.from_file(rel_audio_path_mp3,format='mp3')
        temp.export(rel_audio_path_wav,format='wav')
        print rel_audio_path_mp3
        print rel_audio_path_wav
        #transformer.build(rel_audio_path_mp3, rel_audio_path_wav)
        os.remove(rel_audio_path_mp3)
        
        # split audio into segments, save segments and add to csv record
        article_audio = wave.open(rel_audio_path_wav)
        orig_nChannels = article_audio.getnchannels()
        orig_sampleWidth = article_audio.getsampwidth()
        orig_frameRate = article_audio.getframerate()

        for j, sync_map_segment in enumerate(sync_map_segments):
            # only takes audio segments that are guaranteed to be aligned with text, i.e. not the first 3 and not the last
            print 'Iterating over audio segments: Audio segment ' + str(j) + ' is being created'
            # create path for formatted audio segment
            segment_start, segment_end = sync_map_segment.start_time, sync_map_segment.stop_time
            rel_audio_segment_path_wav = '.' + ''.join(rel_audio_path_wav.split('.')[:-1]) + str(segment_start) + '_' + str(segment_end) + '.wav'
            # convert raw data of segment
            article_audio.setpos(int(segment_start * orig_frameRate))
            chunkData = article_audio.readframes(int((segment_end - segment_start) * orig_frameRate))
            chunkData, _ = audioop.ratecv(chunkData, orig_sampleWidth, orig_nChannels, orig_frameRate, 16000, None)
            # create formatted (framerate = 16000) audio segment and write to disk
            resampled_audio_segment = wave.open(rel_audio_segment_path_wav,'w')
            resampled_audio_segment.setnchannels(orig_nChannels)
            resampled_audio_segment.setsampwidth(orig_sampleWidth)
            resampled_audio_segment.setframerate(16000)
            resampled_audio_segment.writeframes(chunkData)
            resampled_audio_segment.close()
            segment_transcript = sync_map_segment.transcript
            segment_size = path.getsize(rel_audio_segment_path_wav)
            files.append((path.abspath(rel_audio_segment_path_wav).encode('utf-8'), segment_size, segment_transcript.encode('utf-8')))
                    
    #with open(matches_path, 'w') as articles_file_edited:
    #    json.dump(articles_editable, articles_file_edited)
            
    return pandas.DataFrame(data=files, columns=['wav_filename', 'wav_filesize', 'transcript'])
        
class SyncMapSegment(object):
    r"""
    Representation of an individual segment in an STM file.
    """
    def __init__(self, syncmap_line):
        self._id    = syncmap_line['id']
        # record start and end time of segment (in 1/1000 seconds)
        self._start_time  = float(syncmap_line['begin'])
        self._stop_time   = float(syncmap_line['end'])
        #record audio segment duration in seconds
        self._duration = self._stop_time - self._start_time
        #self._transcript  = self._process(syncmap_line['lines'][0].encode('utf-8'))
        self._transcript = syncmap_line['lines'][0].encode('utf-8')

    @property
    def id(self):
        return self._id

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time
    
    @property
    def duration(self):
        return self._duration

    @property
    def transcript(self):
        return self._transcript
    
def parse_syncmap_file(sync_map_path, min_dur_secs, max_dur_secs):
    with open(sync_map_path, 'r') as sync_file:
        sync_map = json.load(sync_file)
        
    syncmap_segments = []
        
    for i,fragment in enumerate(sync_map['fragments']):
        duration = float(fragment['end']) - float(fragment['begin'])
        n_fragments = len(sync_map['fragments'])
        annotation = fragment['lines'][0]
        #syncmap_segments.append(SyncMapSegment(fragment))
        if duration >= float(min_dur_secs) and i not in [0,1,2,3,4,5,6,n_fragments-3,n_fragments-2,n_fragments-1] and duration <= float(max_dur_secs) and duration * 50 > len(annotation) + 10 and len(annotation) != 0:
            print str(duration) + ' is bigger than ' + str(min_dur_secs) + ', smaller than ' + str(max_dur_secs) + ' and is neither at the beginning or end of original audio. Audio segment saved.'
            syncmap_segments.append(SyncMapSegment(fragment))
            global N_OK_AUDIOS
            N_OK_AUDIOS += 1
        elif i in [0,1,2,3,4,5,6, n_fragments-3,n_fragments-2,n_fragments-1]:
            print 'Audio segment is one of the first three of original audio\'s sync map. No audio segment saved.'
            global N_CRIT_POS_AUDIOS
            N_CRIT_POS_AUDIOS += 1
        elif duration < float(min_dur_secs):
            print 'Audiosegment too short: ' + str(duration) + '(audio segment durtion) is smaller than ' + str(min_dur_secs) + '(minimum allowed duration). No audio segment saved.'
            global N_CRIT_SHORT_AUDIOS
            N_CRIT_SHORT_AUDIOS += 1
        elif duration > float(max_dur_secs):
            print 'Audiosegment too long: ' + str(duration) + '(audio segment durtion) is bigger than ' + str(max_dur_secs) + '(maximum allowed duration). No audio segment saved.'
            global N_CRIT_LONG_AUDIOS
            N_CRIT_LONG_AUDIOS += 1
        # catastrophic mismatch, annotation has more characters than audio segment has frames --> breaks tensorflow's ctc implementation
        elif duration * 50 < len(annotation) + 10:
            print 'Catastrophic mismatch: Transcript has ' + str(len(annotation)) + ' characters, but audio has only ' + str(duration*50) + ' frames. No audio segment saved.'
            global N_CRIT_MISMATCH
            N_CRIT_MISMATCH += 1
        elif len(annotation) == 0:
            print 'Empty transcript. No audio segment saved.'
            global N_NO_TRANSCRIPT
            N_NO_TRANSCRIPT += 1
    
    return syncmap_segments

def process_sentence(sentence, transformer):
    # takes a string representing one sentence from a completely UNprocessed
    # article. strips any non-alphanumerical lowercase characters and transforms
    # numericals into words using inflect package
    # lowercase
    sentence_1 = sentence.lower()
    # remove non alpha numerical characters (Except apostrophs and spaces)
    #print 'Creating sentence_2'
    sentence_2 = ''
    for char in sentence_1:
        if char in "01234567890 ,.abcdefghijklmnopqrstuvwxyz'":
            if char == "'":
                print 'Heres an apostroph: ' + sentence_1
            sentence_2 += char
        else:
            sentence_2 += ' '
    #print 'Created sentence_2 ' + sentence_2
    # remove double spaces
    #print 'Creating sentence_3'
    sentence_3 = sentence_2
    for i in range(4):
        sentence_3 = sentence_3.replace('  ',' ')
    #print 'Created sentence_3 ' + sentence_3
    # replace numericals by alphabeticals
    #print 'Creating sentence_4'
    sentence_4 = ''
    index = 0
    while index <= len(sentence_3) - 1:
        current_char = sentence_3[index]
        if current_char not in '0123456789':
            sentence_4 += current_char
            index += 1
        elif current_char in '0123456789':
            converted_numerical, index = convert_numerical(sentence_3[index:],index,transformer)
            sentence_4 += converted_numerical
    #print 'Created sentence_4 ' + sentence_4
    # filter out garbage symbols created by inflect
    #print 'Creating sentence_5'
    sentence_5 = ''
    for char in sentence_4:
        if char not in "abcdefghijklmnopqrstuvwxyz'":
            sentence_5 += ' '
        else:
            sentence_5 += char
    #print 'Created sentence_5 ' + sentence_5
    # filter out double spaces
    #print 'Creating sentence_6'
    sentence_6 = sentence_5
    for i in range(4):
        sentence_6 = sentence_6.replace('  ',' ')
    #print 'Created sentence_6 ' + sentence_6
    
    sentence_final = sentence_6
        
    return sentence_final

def convert_numerical(tail, index, transformer):
    
    char_is_num = True
    position = 0
    numerical = ''
    numerical_tail = ''
    
    while position <= len(tail)-1:
        if tail[position] in '0123456789,.':
            if tail[position] in [',','.']:
                if position < len(tail) - 1:
                    if tail[position+1] in '0123456789':
                        numerical += tail[position]
                    else:
                        numerical_tail = tail[position:position+3]
                        break
                else:
                    numerical_tail = tail[position:position+3]
                    break
            else:
                numerical += tail[position]
            position += 1
        else:
            numerical_tail = tail[position:position+3]
            break
        
    positionals = ['st','nd','rd','th']
    positional_found = False
    
    # convert numerical
    # check for float
    if '.' in numerical:
        converted_numerical = transformer.number_to_words(numerical)
    # check for year
    elif '.' not in numerical and len(numerical) == 4:
        no_group_2 = ['2001','2002','2003','2004','2005','2006','2007','2008','2009']
        if numerical in no_group_2:
            converted_numerical = transformer.number_to_words(numerical)
        elif numerical not in no_group_2:
            if int(numerical) >= 1099 and int(numerical) <= 2050:
                converted_numerical = transformer.number_to_words(numerical,group=2)
            else:
                for positional in positionals:
                    if positional in numerical_tail:
                        # print numerical + 'POSITIONAL'
                        numerical_tail = ''
                        numerical_tail_length = len(positional)
                        converted_numerical = transformer.number_to_words(numerical+positional)
                        positional_found = True
                    if not positional_found:
                        converted_numerical = transformer.number_to_words(numerical)
    else:
        for positional in positionals:
            if positional in numerical_tail:
     #           print numerical + 'POSITIONAL'
                numerical_tail = ''
                numerical_tail_length = len(positional)
                converted_numerical = transformer.number_to_words(numerical+positional)
                positional_found = True
        if not positional_found:
            converted_numerical = transformer.number_to_words(numerical)
        
    # convert units
    unit_converter = {'km ':' kilometres ',
                      'm ':' million ',
                      'bn ':' billion ',
                      'trn':' trillion ',
                      '% ':' percent ',
                      'cm ':' centimetres ',
                      's ':'s '}
    units = ['trn','km ','bn ','cm ','% ','m ','s ']
    unit_found = False
    
    for unit in units:
        if unit in numerical_tail:
            numerical_tail = unit_converter[unit]
            numerical_tail_length = len(unit) # how far ahead did we read into the following text?
            unit_found = True
            
    if not unit_found:
        if not positional_found:
            numerical_tail = ''
            numerical_tail_length = 0 # did not find unit in the following chars, hence jumpt straight back into text
            
    #print numerical_tail_length
    
    converted_total = converted_numerical + numerical_tail
    index += len(numerical) + numerical_tail_length
    
    return converted_total, index

def main():
    _preprocess_data(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))

if __name__ == "__main__":
    main()
