#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:10:00 2020

@author: thorius
"""


from queue import Queue
import numpy as np
import logging
import pyaudio
import time
import collections
from VoiceActivate.tf_stream import StreamActivate
from VoiceControl.tf_stream import StreamControl
import tensorflow as tf



class TfSpeechCommand():
    def __init__(self,
                 path_activate_model = './VoiceActivate/model/keyword_marvin_v3/non_stream',
                 path_control_model = './VoiceControl/model/E2E_1stage_v8/non_stream', 
                 sample_rate = 16000,
                 chunk_duration = 0.08,
                 feed_duration = 1.0,
                 channels = 1,
                 activate_threshold = 0.6,
                 control_threshold = 0.6,
                 control_time = 3):
        
        # create logger to debug
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
         
        # path to activate model
        self.path_activate_model = path_activate_model
        
        # path to control model
        self.path_control_model = path_control_model
        
        
        # sample of audio signal
        self.sample_rate = sample_rate
        
        
        #
        self.device_sample_rate = 44100
        
        
        #chunk_duration -- time in second of a chunk
        self.chunk_duration = chunk_duration
        
        # the threshold for predicting activate word
        self.activate_threshold = activate_threshold
        
        # the threshold for predicting control word
        self.control_threshold = control_threshold
           
        # chanels of audio
        self.channels = channels
        
        # time
        self.control_time = control_time
        
        
        #feed_duration -- time in second of the input to model
        self.feed_duration = feed_duration
        
        
        
        self.chunk_samples = int(self.device_sample_rate * self.chunk_duration)
        self.feed_samples = int(self.device_sample_rate * self.feed_duration)
        
        
        
        # Queue to communiate between the audio callback and main thread
        self.q = Queue()
        
        
        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')
        
        
        # create voice_activate object
        self.voice_activate = StreamActivate(path_model = self.path_activate_model,
                                        sample_rate = self.sample_rate,
                                        chunk_duration = self.chunk_duration,
                                        feed_duration = self.feed_duration,
                                        channels = self.channels,
                                        threshold = self.activate_threshold)
        
        # create voice_control object
        self.voice_control = StreamControl(path_model = self.path_control_model,
                                      sample_rate = self.sample_rate,
                                      chunk_duration = self.chunk_duration,
                                      feed_duration = self.feed_duration,
                                      channels = self.channels,
                                      threshold = self.control_threshold)
        
        
        assert float(self.feed_duration/self.chunk_duration) == float(self.feed_duration/self.chunk_duration)
        
        
        self.stream = True
        
    def run(self):
        
        
        # callback method
        def audio_callback(in_data, frame_count, time_info, status):
            
            # get data from buffer
            data0 = np.frombuffer(in_data, dtype='int16')
            
            # append data from buffer to data
            self.data = np.append(self.data,data0)
            if len(self.data) > self.feed_samples:
                # remove the old data
                self.data = self.data[-self.feed_samples:]
                # Process data async by sending a queue.
                self.q.put(self.data)
            return (in_data, pyaudio.paContinue)
        
        # set up the portaudio system.
        self.audio = pyaudio.PyAudio()
        
        
        
        # open a stream on the device
        self.stream_in = self.audio.open(
            input=True, output=False,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.device_sample_rate,
            frames_per_buffer=self.chunk_samples,
            stream_callback=audio_callback)
        
        
        # the number of predictions in a feed_duration
        size_predicts = int(self.feed_duration / self.chunk_duration)
        
        # the array contains the predictions 
        activate_predictions = np.zeros([size_predicts])
        control_predictions = np.zeros([size_predicts])
        
        try:
            
            while self.stream:
                
                
                # compute activate_predictions in a feed_duration
                for i in range(size_predicts):
                    data = self.q.get()
                    activate_predictions[i] = self.voice_activate.predict(data)
                
                # get best precision and key of word 
                activate_precision, keymax_activate_predictions = self.getBestPredict(activate_predictions, size_predicts)
                
                # if best_precision is greater then threshold for activate word and keymax = 2 - word = marvin
                logging.info("Listening")
                if(activate_precision >= self.activate_threshold and keymax_activate_predictions == 2):
                    logging.info("HI SIR!")
                    
                    message = (time.strftime("%Y-%m-%d %H:%M:%S: ", 
                                            time.localtime(time.time())) + 
                                            self.voice_activate.labels[int(keymax_activate_predictions)] +
                                            "(p: %0.2f)"% (activate_precision))
                    logging.info(message)
                    
                    # set timeout for the process predicting the control word 
                    timeout = time.time() + self.control_time
                    while True:
                        
                        # compute activate_predictions in a feed_duration
                        for i in range(size_predicts):
                            data = self.q.get()
                            control_predictions[i] = self.voice_control.predict(data)
                            
                        # get best precision and key of word
                        control_precision, keymax_control_predictions = self.getBestPredict(control_predictions, size_predicts)
                        
                        
                        if(control_precision >= self.control_threshold):
                            message = (time.strftime("%Y-%m-%d %H:%M:%S:                      ", time.localtime(time.time())) + 
                                                                                                 self.voice_control.labels[int(keymax_control_predictions)] + 
                                                                                                 "(p: %0.2f)"% (control_precision))
                        else:
                            message = (time.strftime("%Y-%m-%d %H:%M:%S:                      ", time.localtime(time.time())) + 
                                                                                                 self.voice_control.labels[1])
                        logging.info(message)
                        if time.time() > timeout:
                            break

                        
        except (KeyboardInterrupt, SystemExit):
            self.stream = False
            # Stop and close the stream 
            self.stream_in.stop_stream()
            self.stream_in.close()
            
            # Terminate the PortAudio interface
            self.audio.terminate()
    
    
    def getBestPredict(self, predictions, size_predicts):
        
        # count the number of each object return a dict
        counter_predictions = collections.Counter(predictions)
        
        # get key of the object with max the number of occurrences
        keymax_predictions = max(counter_predictions, key = counter_predictions.get)
        
        # compute precision
        precision = counter_predictions[keymax_predictions] / size_predicts
        
        return precision, keymax_predictions




def main():
    run_stream = TfSpeechCommand()
    run_stream.run()

if __name__ == '__main__':
    main()
