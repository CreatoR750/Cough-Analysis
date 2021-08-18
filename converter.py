import os
import argparse
import ffmpeg
from pydub import AudioSegment

def convert (A,dirpath):
    for (dirpath, dirnames, filenames) in os.walk(dirpath):
        for filename in filenames:
            if filename.endswith(tuple(A)):

                filepath = dirpath + '/' + filename
                (path, file_extension) = os.path.splitext(filepath)
                file_extension_final = file_extension.replace('.', '')
                try:
                    track = AudioSegment.from_file(filepath,
                            file_extension_final)
                    wav_filename = filename.replace(file_extension_final, 'wav')
                    wav_path = dirpath + '/' + wav_filename
                    print('CONVERTING: ' + str(filepath))
                    file_handle = track.export(wav_path, format='wav')
                    os.remove(filepath)
                except:
                    print("ERROR CONVERTING " + str(filepath))

# Rename folder M4a_files as wav_files