#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:10:00 2020

@author: thorius
"""


import os
import sys
from queue import Queue
import numpy as np
import logging
import pyaudio
import time
import tensorflow as tf 
import json
from tensorflow.keras.models import model_from_json
from kws_streaming.layers.svdf import Svdf
from kws_streaming.layers.speech_features import SpeechFeatures
from kws_streaming.layers.stream import Stream
import collections
from scipy import signal



class StreamActivate():
    def __init__(self, 
                 path_model = './model/keyword_marvin_v2/non_stream', 
                 name_model = 'model_non_stream.json', 
                 sample_rate = 16000,
                 chunk_duration = 0.8,
                 feed_duration = 3.0,
                 channels = 1,
                 threshold = 0.7,
                 pause = False):
        
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        argumentList = sys.argv
   
             
        self.path_model = path_model
        self.name_model = name_model
        self.sample_rate = sample_rate
        
        
        #chunk_duration -- time in second of a chunk
        if(len(argumentList) == 2):
            self.chunk_duration = float(sys.argv[1])
            self.threshold = threshold
        elif(len(argumentList) == 3):
            self.chunk_duration = float(sys.argv[1])
            self.threshold = float(sys.argv[2])
        else:
            self.chunk_duration = chunk_duration
            self.threshold = threshold
           
        # chanels of audio
        self.channels = channels
        
        
        #feed_duration -- time in second of the input to model
        self.feed_duration = feed_duration
        
        #
        self.device_sample_rate = 44100
        
        
        
        self.chunk_samples = int(self.device_sample_rate * self.chunk_duration)
        self.feed_samples = int(self.device_sample_rate * self.feed_duration)
        
        logging.info('chunk samples: ' + str(self.chunk_samples))
        logging.info('feed_samples:' + str(self.feed_samples))
        
        
        # Queue to communiate between the audio callback and main thread
        self.q = Queue()
        
        
        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')
        
        
        
        
        with open(os.path.join(path_model, 'labels.txt'), 'r') as fd:
            labels_txt = fd.read()
        
        self.labels = labels_txt.split()
        
        
        assert float(self.feed_duration/self.chunk_duration) == float(self.feed_duration/self.chunk_duration)
        
        
        self.stream = True
        self.pause = False

        self.model = self.load_model()
        
    def run(self):
        
        
        # callback method
        def audio_callback(in_data, frame_count, time_info, status):
             
            data0 = np.frombuffer(in_data, dtype='int16')
            

            self.data = np.append(self.data,data0)    
            if len(self.data) > self.feed_samples:
                self.data = self.data[-self.feed_samples:]
                # Process data async by sending a queue.
                self.q.put(self.data)
            return (in_data, pyaudio.paContinue)
             
        
        self.audio = pyaudio.PyAudio()
        
        self.stream_in = self.audio.open(
            input=True, output=False,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.device_sample_rate,
            frames_per_buffer=self.chunk_samples,
            stream_callback=audio_callback)
        

        
        try: 
            while self.stream:
                
                
                                
                data = self.q.get()
                print(data.shape)
                preds = self.predictFrames(data)

                new_trigger = self.has_new_triggerword(preds)    
                
                if new_trigger:
                    message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + self.labels[1]
                else:
                    message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + self.labels[0] 
                logging.info(message)
                           
                       
        except (KeyboardInterrupt, SystemExit):
        
            self.stream = False
            # Stop and close the stream 
            self.stream_in.stop_stream()
            self.stream_in.close()
            
            # Terminate the PortAudio interface
            self.audio.terminate()
       
        
    def predict(self, data):
        
        try:
            data = np.array(data, np.float32)
            data = np.expand_dims(data, axis = 0)
            logging.info(data.shape)
            data = signal.resample(data, self.sample_rate, axis = 1)
            logging.info(data.shape)
            
            assert data.shape == (1, 16000)
            # Normalize short ints to floats in range [-1..1).
            #data = data / float(np.max(np.absolute(data)))
            data = np.array(data, np.float32) / 32768.0
            
            predictions = self.model(data)
            
            predicted_labels = np.argmax(predictions)
            
            return predicted_labels
            
        except(AssertionError):
            self.stream = False            
            return -1
    

    def predictFrames(self, data):
        
        
        data = np.array(data, np.float32)
        data = np.expand_dims(data, axis = 0)
        data = signal.resample(data, self.sample_rate * int(self.feed_duration), axis = 1)

        assert data.shape == (1, 48000)

        predictions = self.model.predict(data) > self.threshold
        
        return predictions.reshape(-1)



    def has_new_triggerword(self,predictions):

        chunk_predictions_sample = int(len(predictions) * self.chunk_duration / self.feed_duration)

        chunk_predictions = predictions[-chunk_predictions_sample:]
        
        level = chunk_predictions[0]


        logging.info(level)

        for pred in chunk_predictions:
            if pred > level:
                return True
            else:
                level = pred

        return False

    def load_model(self):
        # load json and create model
        json_file = open(self.path_model + '/' + self.name_model, 'r')

        # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'SpeechFeatures': SpeechFeatures, 
                                                                        'Svdf': Svdf,
                                                                        'Stream': Stream})
        # load weights into new model
        loaded_model.load_weights(self.path_model + '/weights')
        logging.info("Loaded model from disk")

        return loaded_model


def main():
    run_stream = StreamActivate()
    run_stream.run()

if __name__ == '__main__':
    main()
