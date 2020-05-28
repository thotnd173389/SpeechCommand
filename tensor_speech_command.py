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
from play_video import PlayVideo



class TfSpeechCommand():
    def __init__(self,
                 path_activate_model = './VoiceActivate/model/keyword_marvin_v2/non_stream',
                 path_control_model = './VoiceControl/model/E2E_1stage_v8_vl_0_4/non_stream', 
                 sample_rate = 16000,
                 chunk_duration = 0.8,
                 activate_feed_duration = 3.0,
                 control_feed_duration = 1.0,
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
        self.activate_feed_duration = activate_feed_duration
        self.control_feed_duration = control_feed_duration
        
        
        
        self.chunk_samples = int(self.device_sample_rate * self.chunk_duration)
        
        

        
        
        
        # create voice_activate object
        self.voice_activate = StreamActivate(path_model = self.path_activate_model,
                                        sample_rate = self.sample_rate,
                                        chunk_duration = self.chunk_duration,
                                        feed_duration = self.activate_feed_duration,
                                        channels = self.channels,
                                        threshold = self.activate_threshold)
                                    
        
        # create voice_control object
        self.voice_control = StreamControl(path_model = self.path_control_model,
                                      sample_rate = self.sample_rate,
                                      chunk_duration = self.chunk_duration,
                                      feed_duration = self.control_feed_duration,
                                      channels = self.channels,
                                      threshold = self.control_threshold)

        
        
        self.stream = True
        
        self.new_trigger = False

        self.run_vd = PlayVideo()
    def run(self):
        
        
        # callback method
        def audio_callback(in_data, frame_count, time_info, status):
            
            # get data from buffer
            data0 = np.frombuffer(in_data, dtype='int16')
            
            # append data from buffer to data
            self.voice_activate.data = np.append(self.voice_activate.data,data0)

            if len(self.voice_activate.data) > self.voice_activate.feed_samples:
                # remove the old data
                self.voice_activate.data = self.voice_activate.data[-self.voice_activate.feed_samples:]
                # Process data async by sending a queue.
                self.voice_activate.q.put(self.voice_activate.data)
            if self.new_trigger:
                self.voice_control.data = np.append(self.voice_control.data, data0)
                if len(self.voice_control.data) > self.voice_control.feed_samples:
                    # remove the old data
                    self.voice_control.data = self.voice_control.data[-self.voice_control.feed_samples:]
                    # Process data async by sending a queue.
                    self.voice_control.q.put(self.voice_control.data)
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
    
        control_list, channel_list = self.run_vd.setDefaultList()
        
        self.run_vd.settings()

        try:
            
            while self.stream:
                data = self.voice_activate.q.get()
                preds = self.voice_activate.predictFrames(data)

                self.new_trigger = self.voice_activate.has_new_triggerword(preds)    
                
                if self.new_trigger:
                    message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + self.voice_activate.labels[1]
                    logging.info(message)

                    #self.new_trigger = False

                    self.voice_activate.q.queue.clear()

                    # set timeout for the process predicting the control word
                    #time.sleep(3) 
                    timeout = time.time() + self.control_time


                    while True:
                        data = self.voice_control.q.get()
                        control_predicted_label = self.voice_control.predict(data)

                        new_keyword = self.voice_control.has_new_keyword(control_predicted_label)
                        message = (time.strftime("%Y-%m-%d %H:%M:%S:                      ", time.localtime(time.time())) + 
                                                                                            self.voice_control.labels[control_predicted_label])
                        logging.info(message)

                        if new_keyword:
                            
                            current_channel = self.run_vd.getIndexChannel(channel_list)
                            
                            
                            control_list, channel_list = self.run_vd.setControlList(control_list, channel_list, control_predicted_label)
                            
                            new_channel = self.run_vd.getIndexChannel(channel_list)
                            logging.info("channel: " + str(new_channel))
                            
                            if control_list[1] == 1:
                                self.run_vd.stopPlayVideo()
                            elif (current_channel != new_channel and control_list[0] == 1):
                                self.run_vd.stopPlayVideo()
                                self.run_vd.setChannel(new_channel)
                                
                                self.run_vd.setConfig()
                                self.run_vd.startPlayVideo()
                            
                            self.voice_control.q.queue.clear()
                            break
                            

                        if time.time() > timeout:
                            self.voice_control.q.queue.clear()
                            break

                else:
                    message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + self.voice_activate.labels[0] 
                    logging.info(message)

                        
        except (KeyboardInterrupt, SystemExit):
            self.stream = False
            # Stop and close the stream 
            self.stream_in.stop_stream()
            self.stream_in.close()
            
            # Terminate the PortAudio interface
            self.audio.terminate()
    
    




def main():
    run_stream = TfSpeechCommand()
    run_stream.run()

if __name__ == '__main__':
    main()
