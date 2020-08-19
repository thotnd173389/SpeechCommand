from py_irsend import irsend
import time

#print(irsend.list_remotes())
#print(irsend.list_codes('SHARP'))
#Convert predict labels to button corresponding
def label_to_button(idx):
    switch={
        2:'KEY_POWER',
        3:'KEY_POWER',
        4:'KEY_VOLUMEUP',
        5:'KEY_VOLUMEDOWN',
        6:'KEY_0',
        7:'KEY_1',
        8:'KEY_2',
        9:'KEY_3',
        10:'KEY_4',
        11:'KEY_5',
        12:'KEY_6',
        13:'KEY_7',
        14:'KEY_8',
        15:'KEY_9',
        16:'KEY_OK'
        }
    return switch.get(idx, "No exist")

#Send IR signal to TV
def ir_send(btn):
    
    if (btn == 'KEY_VOLUMEUP') or (btn == 'KEY_VOLUMEDOWN'):
        irsend.send_start('SHARP', btn)
        time.sleep(1)
        irsend.send_stop('SHARP', btn)
    
    irsend.send_start('SHARP', btn)
    time.sleep(0.5)
    irsend.send_stop('SHARP', btn)
       
