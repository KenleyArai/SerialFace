
from SerialFace.SerialFace import SerialFace

face = SerialFace("SerialFace/haarcascade_frontalface_default.xml",
                  "/dev/ttyAMA0",
                  9600,
                  timeout=1)
face.run()
