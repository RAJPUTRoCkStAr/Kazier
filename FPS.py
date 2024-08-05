import time
import cv2
class FPS:
    def __init__(self):
        self.ptime = time.time()
        self.frame_count = 0
    def showfps(self, display=None, writetext=True, text_pos=(10, 70),
               Fstyle =cv2.FONT_HERSHEY_COMPLEX ,
               tcolor=(255, 0, 0),
               fscale=1,fthickness=2,Sverbose=True):
        self.frame_count += 1
        cTime = time.time()
        fps = 1 / (cTime - self.ptime)
        self.ptime = cTime
        if Sverbose:
            print(f"FPS: {fps:.2f}")
        if writetext and display is not None:
            cv2.putText(display, f"FPS: {int(fps)}", 
                        text_pos,Fstyle, fscale, 
                        tcolor, fthickness)
        return fps
def main():
    fps_counter = FPS()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fps = fps_counter.showfps(frame, writetext=True, text_pos=(10, 50),
                                fthickness=2,tcolor=(0,255,250),
                                Fstyle=cv2.FONT_HERSHEY_DUPLEX,fscale=2,)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
