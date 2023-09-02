import wave
from dataclasses import dataclass, asdict
import whisper
import pyaudio
from threading import Thread
import numpy as np
import time
import sys
import re

import audioop
import math
from collections import deque
from queue import Queue

def Processing(model, recordings: Queue):
    while True:
        while not recordings.empty():
            result = model.transcribe(recordings.get(), language="polish")
            result = result['text']
            print(result)
            if(re.search("Exit", result) == "Exit"):
                print("My nutz in yo face")
        time.sleep(0.5)

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 44100
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Recorder:

    def __init__(self, stream_params: StreamParams, save_path: str) -> None:
        self.index = 0
        #Setting parameters
        self.save_path = save_path + str(self.index) + ".mp3"
        self.stream_params = stream_params
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        #Creating sound file
        self._wav_file = wave.open(self.save_path, "wb")
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)


    def createNewFile(self):
            self.index+=1
            #max number of files 10
            if self.index >= 10:
                self.index = 0
            self.save_path = self.save_path[:6] + str(self.index) + ".mp3"
            self._wav_file.close()
            self._wav_file = wave.open(self.save_path, "wb")
            self._wav_file.setnchannels(self.stream_params.channels)
            self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
            self._wav_file.setframerate(self.stream_params.rate)

    def record(self, recordings) -> None:
        
        print("Start recording...")
        #to be changed
        silenceLimit = 0.1
        silence_threshold=2000
        silenceI = 0

        listen = True
        startedRecording = False
        prev_audio_data = deque(maxlen=int(0.5 * self.stream_params.rate/self.stream_params.frames_per_buffer))
        slid_window = deque(maxlen=int(1 * self.stream_params.rate/self.stream_params.frames_per_buffer))
        
        #Capturing sound data
        while listen:
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            slid_window.append(math.sqrt(abs(audioop.avg(audio_data, 4))))


            #add to the file missing audio from short perioid of silence
            if (sum([x > silence_threshold for x in slid_window]) > 0):
                self._wav_file.writeframes(b''.join(prev_audio_data))
                prev_audio_data.clear()
            #if not silent record sound
            while(sum([x > silence_threshold for x in slid_window]) > 0):
                #Record sound and write to file
                audio_data = self._stream.read(self.stream_params.frames_per_buffer)
                slid_window.append(math.sqrt(abs(audioop.avg(audio_data, 4))))
                silenceI = 0
                startedRecording = True
                self._wav_file.writeframes(audio_data)

            #else count seconds of silence    
            #collect sound from perioids of "silence"
            prev_audio_data.append(audio_data)
                
            silenceI+=1
            silenceLen = round(silenceI/(self.stream_params.rate/self.stream_params.frames_per_buffer), 2)
            #print("Seconds of silence: ", silenceLen)

            #If long silence after started recording: reset, sent recording to transcription and go to next file
            if(silenceLen >= silenceLimit and startedRecording == True):
                prev_audio_data.clear()
                #sent to transcription and start working on next file
                recordings.put(self.save_path)
                self.createNewFile()
                startedRecording = False
            

        #Terminate stream
        self._wav_file.close()
        self._stream.close()
        self._pyaudio.terminate()
        
        print("Stop recording")


class Controler():

    def __init__(self, model: str):
        stream_params = StreamParams()
        self.model = whisper.load_model(model)
        self.recorder = Recorder(stream_params, "output")
        self.recordings = Queue()

    def startRecording(self):
        self.recordT = Thread(target=self.recorder.record, args=[self.recordings])
        self.recordT.start()
        self.startTranscription()
        self.recordT.join()

    def startTranscription(self):
        self.transcriptionT = Thread(target=Processing, args=[self.model, self.recordings])
        self.transcriptionT.start()
            


#========================================== Main ============================================

controler = Controler("small")
controler.startRecording()
print("END")
