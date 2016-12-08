import serial
from .Face import Face


class SerialFace(object):

    signals = ['n', 'u']
    ERROR = 'E'

    def __init__(self, path, *args, **kwargs):
        self._ser = serial.Serial(*args, **kwargs)
        self._face = Face(path)

        signal_functions = [self._face.new_pass,
                            self._face.can_unlock]

        self.switch = dict(zip(self.signals, signal_functions))

    def __del__(self):
        self._ser.close()

    def run(self):
        while 1 and self._ser.isOpen():
            message = self._ser.read() # Read 1 byte of data
            if message and message in self.signals:
                self.switch[message]()
            else:
                self._ser.write(self.ERROR)
