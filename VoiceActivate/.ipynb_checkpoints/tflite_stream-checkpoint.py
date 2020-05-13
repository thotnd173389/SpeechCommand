import os
import sys
from queue import Queue
import numpy as np
import pyaudio
import time
import tensorflow.compat.v1 as tf



class TFLiteStream():
    def __init__(self, 
                 path_model = './model/E2E_1stage_v5/tflite_non_stream', 
                 name_model = 'non_stream.tflite', 
                 sample_rate = 16000,
                 chunk_duration = 0.75,
                 feed_duration = 1.0,
                 channels = 1):
        
        argumentList = sys.argv
   
             
        self.path_model = path_model
        self.name_model = name_model
        self.sample_rate = sample_rate
        
        
        #chunk_duration -- time in second of a chunk
        if(len(argumentList) == 2):
            self.chunk_duration = float(sys.argv[1])
        else:
            self.chunk_duration = chunk_duration
           
        # chanels of audio
        self.channels = channels
        
        
        #feed_duration -- time in second of the input to model
        self.feed_duration = feed_duration
        
        
        
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.feed_samples = int(self.sample_rate * self.feed_duration)
        
        
        
        # Queue to communiate between the audio callback and main thread
        self.q = Queue()
        
        
        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')
        
        
        
        
        with open(os.path.join('./model', 'labels.txt'), 'r') as fd:
            labels_txt = fd.read()
        
        self.labels = labels_txt.split()
        
        
        assert float(self.feed_duration/self.chunk_duration) == float(self.feed_duration/self.chunk_duration)
        
    def run(self):
        
        # Start a new TensorFlow session.
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
    
        
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
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_samples,
            stream_callback=audio_callback)       
        
        try: 
            while True:
                
                data = self.q.get()
                                        
                predictions = self.predict(data)
            
                message = time.strftime("%Y-%m-%d %H:%M:%S: ", time.localtime(time.time())) + predictions
                
                print(message)
                       
        except (KeyboardInterrupt, SystemExit):
            # Stop and close the stream 
            self.stream_in.stop_stream()
            self.stream_in.close()
            
            # Terminate the PortAudio interface
            self.audio.terminate
       
    
    def predict(self, data):
        
        try:
            data = np.array(data, np.float32)
            data = np.expand_dims(data, axis = 0)
            assert data.shape == (1, 16000)
            # Normalize short ints to floats in range [-1..1).
            #data = data / float(np.max(np.absolute(data)))
            data = np.array(data, np.float32) / 32768.0
        
        
            # prepare TFLite interpreter
            with open(os.path.join(self.path_model, self.name_model), 'rb') as f:
                model_content = f.read()
            
            interpreter = tf.lite.Interpreter(model_content=model_content)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
                        
            padded_input = np.zeros((1, 16000), dtype=np.float32)
            padded_input[:, :data.shape[1]] = data
            
            # set input audio data (by default data at index 0)
            interpreter.set_tensor(input_details[0]['index'], padded_input.astype(np.float32))
            
            # run inference
            interpreter.invoke()
            
            # get output: classification
            out_tflite = interpreter.get_tensor(output_details[0]['index'])
            
            out_tflite_argmax = np.argmax(out_tflite)
            
        
            
            return self.labels[out_tflite_argmax] 
            
        except(AssertionError):            
            return "Error"     
        
        
        


def main():
    run_stream = TFLiteStream()
    run_stream.run()

if __name__ == '__main__':
    main()
