import cv2
from cvzone.HandTrackingModule import HandDetector
import sys
import webcam

class Controle:

    gc_mode = 0
    cap = None

    def __init__(self):
        Controle.gc_mode = 1
        Controle.cap = cv2.VideoCapture(0)

    def start(self):
        while Controle.cap.isOpened() and Controle.gc_mode:
            # Récupération de l'image de la webcam
            success, img = Controle.cap.read()

            detector = HandDetector(detectionCon=0.8, maxHands=1)

            buttonRecFac = Button((100, 50), 500, 100, "Reconnaissance faciale")
            buttonQuit = Button((700, 50), 300, 100, "Quitter")
            buttonNew = Button((700, 200), 300, 100, "Nouveau Bouton")

            if not success:
                print("Ignoring empty camera frame.")
                continue

            img = cv2.resize(img, (1024, 720), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
            img = cv2.flip(img, 1)
            #img = cv2.namedWindow("Image")
            hands, img_hand = detector.findHands(img, flipType=False)

            # Afficher les boutons
            buttonRecFac.draw(img_hand)
            buttonQuit.draw(img_hand)

            if hands:
                # Main
                lmList = hands[0]['lmList']
                length, _, img_dist = detector.findDistance(lmList[8], lmList[12], img) 
                x, y = lmList[8]
                if length < 50:
                    if buttonRecFac.checkClick(x, y, img_dist):
                        webcam4.launch_algo()
                        #Controle.cap.release()
                        cv2.imshow("Image", img)
                        cv2.waitKey(100)
                        Controle.cap.release()
                        cv2.destroyAllWindows()
                    if buttonQuit.checkClick(x, y, img_dist):
                        sys.exit()

            cv2.imshow("Image", img)
            cv2.waitKey(1)
        
        #Controle.cap.release()
        #cv2.destroyAllWindows()

class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height), 
            (225, 225, 225), cv2.FILLED)

        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height),
            (50, 50, 50), 3)
        
        cv2.putText(img, self.value, (self.pos[0]+ 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,
            2, (50, 50, 50), 2)

    def checkClick(self, x, y, img):
        if self.pos[0] < x < self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height), 
                (255, 255, 255), cv2.FILLED)

            cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height),
                (50, 50, 50), 3)
        
            cv2.putText(img, self.value, (self.pos[0]+ 20, self.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 0), 5)

            return True
        else:
            return False

    def drawNew(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height), 
            (225, 225, 225), cv2.FILLED)
        
        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height),
            (50, 50, 50), 3)
    
        cv2.putText(img, self.value, (self.pos[0]+ 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,
            2, (50, 50, 50), 2)


