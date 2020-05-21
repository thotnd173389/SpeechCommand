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



class StreamControl():
    def __init__(self, 
                 path_model = './model/E2E_1stage_v8/non_stream', 
                 name_model = 'model_non_stream.json', 
                 sample_rate = 16000,
                 chunk_duration = 0.25,
                 feed_duration = 1.0,
                 channels = 1,
                 threshold = 0.5,
                 time_out = 8):
        
        
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
        
        # times for ru   
        # chanels of audio
        self.channels = channels
        
        
        #feed_duration -- time in second of the input to model
        self.feed_duration = feed_duration
        
        self.device_sample_rate = 44100
        
        self.chunk_samples = int(self.device_sample_rate * self.chunk_duration)
        self.feed_samples = int(self.device_sample_rate * self.feed_duration)
        
        
        
        # Queue to communiate between the audio callback and main thread
        self.q = Queue()
        
        
        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')
        
        
        
        
        with open(os.path.join(path_model, 'labels.txt'), 'r') as fd:
            labels_txt = fd.read()
        
        self.labels = labels_txt.split()
        
        
        assert float(self.feed_duration/self.chunk_duration) == float(self.feed_duration/self.chunk_duration)
        
        
        self.stream = True

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
                   
        size_predicts = int(int(self.feed_duration / self.chunk_duration))
        predictions = np.zeros([size_predicts])
        
        try: 
            while self.stream:
            
                current_time = time.time()
                                
                for i in range(size_predicts):
                    data = self.q.get()
                    predictions[i] = self.predict(data)
                    
                counter_predictions = collections.Counter(predictions)
                
                predictions[:size_predicts] = 0
                
                keymax_predictions = max(counter_predictions, key = counter_predictions.get)
                
                precision = counter_predictions[keymax_predictions] / size_predicts
                
                if(precision >= self.threshold):
                    message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + self.labels[int(keymax_predictions)] + "(p: %0.2f)"% (precision)
                else:
                    message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + self.labels[1] 
                logging.info(message)
                
                lastprocessing_time = time.time()
                remain_time = lastprocessing_time - current_time
                if(remain_time < self.feed_duration):
                    time.sleep(remain_time)                
                       
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
            
            data = signal.resample(data, self.sample_rate, axis = 1)
            
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
        
    def load_model(self):
        # load json and create model
        json_file = open(self.path_model + '/' +  self.name_model, 'r')

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
    run_stream = StreamControl()
    run_stream.run()

if __name__ == '__main__':
    main()
