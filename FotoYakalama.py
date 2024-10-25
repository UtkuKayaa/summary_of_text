import cv2

class FotoYakalama:
    def __init__(self, img=None):
        self.img = img

    def foto_yakalama(self):
        cap = cv2.VideoCapture(0)  # Kamera açılıyor
        while True:
            ret, frame = cap.read() 
            if ret:
                cv2.imshow("Kamera Görüntüsü", frame)  
        
            key = cv2.waitKey(1) & 0xFF  
            if key == ord('c'):  
                cv2.imwrite("captured_photo.jpg", frame)  
                print("Fotoğraf çekildi ve kaydedildi: captured_photo.jpg")
            elif key == ord('q'):  
                break
        cap.release()  
        cv2.destroyAllWindows()  
        return frame  
