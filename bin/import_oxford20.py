# To be run from the DeepSpeech root dir
# creates the subdir ./data/oxford_20 if not alreday existent

import tarfile
import sys
import os
from tensorflow.python.platform import gfile

def _maybe_download(data_dir,zip_file):
    # check for data folder
    if not gfile.Exists(data_dir):
        os.mkdir(data_dir)
    # check for oxford zip file
    if not gfile.Exists(os.path.join(data_dir,zip_file)):
        os.system('gsutil cp gs://ais-data/speech2text/economist/'+zip_file+' '+data_dir)

def _maybe_extract(data_dir, zip_file, extracted_data):
    # check if zip file has been extracted; if not, extract
    extracted_data_path = os.path.join(data_dir, extracted_data)
    if not gfile.Exists(extracted_data_path):
        zip_path = os.path.join(data_dir, zip_file)
        tar = tarfile.open(zip_path)
        tar.extractall(data_dir)
        tar.close()

def main():
    data_dir = sys.argv[1]
    ZIP_FILE = 'OXFORD20.tar.gz'
    EXTRACTED_DATA = 'OXFORD20'
    _maybe_download(data_dir, ZIP_FILE)
    _maybe_extract(data_dir, ZIP_FILE, EXTRACTED_DATA)
    print 'The oxford_20 dev data has been downloaded and extracted into ' + data_dir

if __name__ =='__main__':
    main()
