import time
import math
from openal.audio import SoundSink, SoundSource
from openal.loaders import load_wav_file

if __name__ == "__main__":
    sink = SoundSink()
    sink.activate()

    """
    Source Pos x,y,z

    x (horizontal axis)
    -ve left | +ve right

    y(vertical axis)

    """
    source = SoundSource(position=[0, 0, 0],
                         velocity=[0, 0, 0],)

    source.looping = True
    data = load_wav_file("beep.wav")
    source.queue(data)
    sink.play(source)
    t = 0

    for i in range(10):
        sink.update()
        time.sleep(0.1)

    while True:
        x_pos = 5*math.sin(math.radians(t))
        source.position = [source.position[0], source.position[1], source.position[2]]
        sink.update()
        print("playing at %r" % source.position)
        time.sleep(0.1)
        t += 5
