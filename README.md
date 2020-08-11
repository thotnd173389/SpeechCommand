# SpeechCommand

---

# Description
  The project aims to use keyword recognition streaming in a real-time offline embedded system using RaspberryPi3.




## How to run


### Requirements

```shell script
pip install -U -r requirement.txt
```


### Keyword recognition streaming

- Run with Tensorflow

```shell script
python tensor_speech_command.py
```

- Run with TensorflowLite

```shell script
python speech_command.py
```
- The keyword recognizing process has two stages.

#### Stage 1: Recognize the trigger keyword.
     - The trigger word is Marvin.
     - When you speak Marvin, if the system recognizes the trigger word, the chime sound is played.
     - Move to stage 2 
      

#### Stage 2: Recognize the command keywords.
     - The command keywords are zero, one, two, three, four, five, six, seven, eight, nine, up, down, on, off.
     - The system will recognize the command keyword in control_time s.
     - If the system recognizes the command keyword, it will execute the corresponding command for that keyword.
     - Until the time for this process is greater than control time, it will execute stage 1.
