#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:07:50 2020

@author: thorius
"""
import vlc
import time
import logging


class PlayVideo():
    def __init__(self, 
                 path_video_demo_folder = './video_demo/'):
        
        # create logger to debug
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.path_video_demo_folder = path_video_demo_folder
        
    def settings(self):
        self.instance_vlc = vlc.Instance()
        self.media_player = self.instance_vlc.media_player_new()
    
    
    def setChannel(self, index_channel):
        
        self.Media = self.instance_vlc.media_new(self.path_video_demo_folder  + str(index_channel) + '.mp4')
        self.Media.get_mrl()
    
    def getIndexChannel(self, channel_list):
        index_channel = -1
        for i in range(len(channel_list)):
            if channel_list[i] == 1:
                index_channel = i
        return index_channel
     
    
    def setConfig(self, enable_fullscreen = False):
        
        self.media_player.set_media(self.Media)
        self.media_player.set_fullscreen(enable_fullscreen)
        
        
            
    def startPlayVideo(self):
        self.media_player.play()
        
        
    def stopPlayVideo(self):
        self.media_player.stop()


    def mute(self):
        self.media_player.audio_set_volume(0)

        
    def unMute(self):
        self.media_player.audio_set_volume(50)
        
        
    def setControlList(self, control_list, channel_list, keymax_control_predictions):
        
         
        
        currentChannel = self.getIndexChannel(channel_list)
        
        
        if keymax_control_predictions == 2:
            control_list[0] = 1
        elif keymax_control_predictions == 3:
            control_list[0] = 0
            
        elif keymax_control_predictions == 4:
            channel_list[currentChannel] = 0
            
            if currentChannel == 9:
                channel_list[0] = 1
            else:
                channel_list[currentChannel+1] = 1
                
        elif keymax_control_predictions == 5:
            channel_list[currentChannel] = 0
            channel_list[currentChannel-1] = 1
        elif keymax_control_predictions > 5 and currentChannel + 6 != keymax_control_predictions:
            channel_list[currentChannel] = 0
            channel_list[keymax_control_predictions-6] = 1
        
        return control_list, channel_list
    
    
    
    def setDefaultList(self):
        control_list = [1]
        channel_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        
        return control_list, channel_list
        
        
class PlayAudio():
    def __init__(self, path_audio = './audio/activate_0.wav'):
        self.path_audio = path_audio
        self.player = vlc.MediaPlayer(self.path_audio)
    
    def play_audio(self):
        self.player.play()
        time.sleep(1)
        self.player.stop()


    def stop_audio(self):
        self.player.stop()

            
if __name__ == '__main__':
    py_vd = PlayVideo()
    py_vd.settings()