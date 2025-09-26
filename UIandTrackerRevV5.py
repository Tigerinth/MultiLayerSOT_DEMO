import cv2
import numpy as np
import time
import psutil
import threading
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
'''

MEP Detection
Multiple Edge and Pattern Detection -> 
bu isim tasarlanan algoritmalar sonrası yapılan işlemler baz alınarak kısa bir adlandırma olarak uydurulmuştur 

'''
#INITIALIZE

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.4 # YOLO guven threshold takip ayri

cpu_usage = 0.0
def monitor_cpu_usage():
    global cpu_usage
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)*10
        print(f"CPU Kullanimi: {cpu_usage}%") #for debug
        time.sleep(1)

cpu_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
cpu_thread.start()


cap = cv2.VideoCapture(0)
prevTime = time.time()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

panelWidth = 320
panelHeight = 640
border = 10
buttonSpace = 10

btnrectWidth = (panelWidth - 2 * border - buttonSpace) // 2
btnrectHeight = (panelHeight - 2 * border - buttonSpace * 3) // 4

MENU = [f'func {i+1}' for i in range(8)]
MENU[0] = "HUD"
MENU[1] = "BOOST"
MENU[2] = "ROI"
MENU[3] = "Track"
MENU[4] = "LINE"
MENU[5] = "IMGTest"
MENU[6] = "Metrics"
MENU[7] = "EXIT"

panel = np.zeros((panelHeight, panelWidth, 3), dtype=np.uint8)
panel[:] = (100, 100, 100)

selectedButton = 0

results = None
histogramImg = None
prev_gray = None

ROIBlink = False
lockedOK = False
timeNow = 0.0
lastTime = 0.0
trackCenterX = 0.0
trackCenterY = 0.0
topLeft = None
bottomRight = None
YOLOAndTrackerPaired = False

#pre settings
ROIEnabled = False
HUDEnabled = True
BOOSTEnabled = False
MetricsEnabled = True
LineEnabled = True
IMGTestEnabled = False
TrackMode = 0
exitFlag = False
frame_x = 1280 #default
frame_y = 720 #default
hovered_index = -1  # Baslangic butonu default

#menuyu olustr
rectangles = []
for i in range(4):
    for j in range(2):
        x1 = j * (btnrectWidth + buttonSpace) + border
        y1 = i * (btnrectHeight + buttonSpace) + border
        x2 = x1 + btnrectWidth
        y2 = y1 + btnrectHeight
        rectangles.append(((x1, y1, x2, y2), MENU[i * 2 + j]))

#kontrol paneli mouse koordinat fonksiyonu
def panelOnMouseClick(event, x, y, flags, param):
    global selectedButton
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (coords, label) in enumerate(rectangles):
            x1, y1, x2, y2 = coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                #print(i + 1) #for debug
                selectedButton = i+1

def panelOnMouseEvent(event, x, y, flags, param):
    global hovered_index
    global selectedButton
    # Hover
    if event == cv2.EVENT_MOUSEMOVE:
        hovered_index = -1  # sifirlama
        for i, (coords, label) in enumerate(rectangles):
            x1, y1, x2, y2 = coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                hovered_index = i

    # Selection
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (coords, label) in enumerate(rectangles):
            x1, y1, x2, y2 = coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                #print(i + 1)  # for debug
                selectedButton = i+1


cv2.namedWindow("Kontrol Paneli")
cv2.setMouseCallback("Kontrol Paneli", panelOnMouseEvent)

#cap = cv2.VideoCapture(0)
#prev_time = time.time()

roiSelected = False
template_MEP = None
FilteredFrame = None
rotated_templates = []
currentPos = None
trackerThreshold = 0.4

trackingLost = True
TrackFrameLostCounter = 0
TrackFrameLostThreshold = 10

#ORB Keyfeatures ozellik cikarimi icindir
#Parametreler Yogun islem yapmasi icin kullanilmistir
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=10)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# for Mouse callback
clickPoint = None
frameForClick = None

def rotateTemplate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def computeHistogramScore(template, search_roi):
    hsvTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    hsvSearch = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)

    histTemplate = cv2.calcHist([hsvTemplate], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_search = cv2.calcHist([hsvSearch], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(histTemplate, histTemplate, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_search, hist_search, 0, 1, cv2.NORM_MINMAX)

    return cv2.compareHist(histTemplate, hist_search, cv2.HISTCMP_CORREL)

def mouseCallback(event, x, y, flags, param):
    global clickPoint, roiSelected, template_MEP, rotated_templates, ROIBGRTemplate, kp_template, des_template, currentPos, trackingLost

    if event == cv2.EVENT_LBUTTONDOWN:
        clickPoint = (x, y)

        # ROI size
        ROISize = 150
        half_size = ROISize // 2

        # Kare ROI koordinatlari
        x1 = max(x - half_size, 0)
        y1 = max(y - half_size, 0)
        x2 = x1 + ROISize
        y2 = y1 + ROISize

        #  frame sınırlarını aşarsa düzelt
        frame_h, frame_w = frameForClick.shape[:2]
        if x2 > frame_w:
            x2 = frame_w
            x1 = x2 - ROISize
        if y2 > frame_h:
            y2 = frame_h
            y1 = y2 - ROISize

        ROIbgr = frameForClick[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(ROIbgr, cv2.COLOR_BGR2GRAY)

        # TEMPLATE OLUSTURMA
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        canny = cv2.Canny(roi_gray, 100, 200)
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)

        grayNorm = cv2.normalize(roi_gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        sobelxNorm = cv2.normalize(np.abs(sobelx), None, 0, 1, cv2.NORM_MINMAX)
        sobelyNorm = cv2.normalize(np.abs(sobely), None, 0, 1, cv2.NORM_MINMAX)
        cannyNorm = cv2.normalize(canny.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        laplacianNorm = cv2.normalize(np.abs(laplacian), None, 0, 1, cv2.NORM_MINMAX)

        template_MEP = (0.2 * grayNorm +
                             0.3 * sobelxNorm +
                             0.3 * sobelyNorm +
                             0.1 * cannyNorm +
                             0.1 * laplacianNorm)
        template_MEP = (template_MEP * 255).astype(np.uint8)

        #rotated_templates = [rotateTemplate(template_MEP, angle) for angle in [0]]

        ROIBGRTemplate = ROIbgr.copy()
        kp_template, des_template = orb.detectAndCompute(template_MEP, None)

        roiSelected = True
        trackingLost = False
        currentPos = np.array([[x, y]], dtype=np.float32)

        print(f"ROI merkez = ({x},{y})")

# Mouse callback to frame
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouseCallback)

while True:
    #ret, frame = cap.read()
    #if not ret:
    #    break

    # FRAME READING
    if not IMGTestEnabled:
        ret, frame = cap.read()
        frame_x = frame.shape[0]
        frame_y = frame.shape[1]
    else:
        frame = cv2.imread("IMGTest.png")
        frame = cv2.resize(frame, (frame_y, frame_x))
        ret = frame is not None
    if not ret:
        break

    frameForClick = frame.copy()  # Mouse callback icin frame guncelliyorum

    key = cv2.waitKey(1) & 0xFF

    # CONTROL PANEL

    panel[:] = (50, 50, 75)  # PANEL COLOR


    for i, ((x1, y1, x2, y2), label) in enumerate(rectangles):
        if i == hovered_index:  # HOVER ON
            color = (255, 50, 50)
            text_color = (255, 50, 50)
        else:
            color = (200, 200, 200)  # HOVER OFF
            text_color = (200, 200, 200)

        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
        text_x = x1 + (btnrectWidth - text_size[0]) // 2
        text_y = y1 + (btnrectHeight + text_size[1]) // 2
        cv2.putText(panel, label, (text_x, text_y), font, 0.8, text_color, 2)
    '''
    for (x1, y1, x2, y2), label in rectangles:
        cv2.rectangle(panel, (x1, y1), (x2, y2), (200, 200, 200), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
        text_x = x1 + (rect_width - text_size[0]) // 2
        text_y = y1 + (rect_height + text_size[1]) // 2
        cv2.putText(panel, label, (text_x, text_y), font, 0.8, (200, 200, 200), 2)
    '''

    # ROI BLINK

    lockedOK = False
    timeNow = time.time()
    if timeNow - lastTime > 0.5:
        ROIBlink = not ROIBlink
        lastTime = timeNow

    # selectROI algoritması pek hos calismiyor mouse tıklamasini mecburen tekrar koydum.
    # Eğer takip kaybolduysa yine mouse ile tıklaniyor zaten takip disina düser.

    #METRICS

    current_time = time.time()
    elapsed_time = current_time - prevTime
    prevTime = current_time

    delay_ms = elapsed_time * 1000

    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    fps = max(0,(min(fps,32)))


    text = f"Delay: {delay_ms:.2f} ms | FPS: {fps:.2f} | CPU: %{cpu_usage:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    generalFontScale = 0.7
    text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = frame.shape[0] - 10

    #MENU FONKSIYONLARI BURADA YAZILIYOR SADECE 2 FONKSIYON VAR SUAN (edit: yapildi)

    #MENU STATUS
    # match case py3.10 da cikmis ilk deneme
    match selectedButton:
        case 1:
            HUDEnabled = not HUDEnabled
        case 2:
            BOOSTEnabled = not BOOSTEnabled
        case 3:
            ROIEnabled = not ROIEnabled
        case 4:
            print("track")
            TrackMode +=1
            if TrackMode >= 2:
                TrackMode = 0
            print(TrackMode)
        case 5:
            print("LINE")
            LineEnabled = not LineEnabled
        case 6:
            print("IMGTest")
            IMGTestEnabled = not IMGTestEnabled
        case 7:
            print("Metrics")
            MetricsEnabled = not MetricsEnabled
        case 8:
            print("exit")
            exitFlag = True

    #ROI

    #MERKEZ ROI ve TRACKER ROI BELIRLENEN YER
    fixedROISize = 200
    fixedROIX = (frame.shape[1] - fixedROISize) // 2
    fixedROIY = (frame.shape[0] - fixedROISize) // 2

    if BOOSTEnabled:
        results = model(frame)

    if roiSelected and not trackingLost:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TEMPLATE MATCHLENMESI ICIN PENCERE

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        canny = cv2.Canny(gray, 100, 200)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        grayNorm = cv2.normalize(gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        sobelxNorm = cv2.normalize(np.abs(sobelx), None, 0, 1, cv2.NORM_MINMAX)
        sobelyNorm = cv2.normalize(np.abs(sobely), None, 0, 1, cv2.NORM_MINMAX)
        cannyNorm = cv2.normalize(canny.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        laplacianNorm = cv2.normalize(np.abs(laplacian), None, 0, 1, cv2.NORM_MINMAX)

        combined = (0.2 * grayNorm +
                    0.3 * sobelxNorm +
                    0.3 * sobelyNorm +
                    0.1 * cannyNorm +
                    0.1 * laplacianNorm)
        combined = (combined * 255).astype(np.uint8)

        bestVal = -1
        bestLoc = None

        #korelasyon islemlerinin asil yapildigi yer burasi
        #Korelasyon işlemi önce NumPy ile yazildi fakat opencv nin algoritmaları kat kat daha optimize çalışıyor
         #for temp in rotated_templates:
        res = cv2.matchTemplate(combined, template_MEP, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > bestVal:
            bestVal = max_val
            bestLoc = max_loc

        topLeft = bestLoc
        bottomRight = (topLeft[0] + template_MEP.shape[1], topLeft[1] + template_MEP.shape[0])

        # Boundary control
        frame_height, frame_width = frame.shape[:2]
        if topLeft[0] < 0: topLeft = (0, topLeft[1])
        if topLeft[1] < 0: topLeft = (topLeft[0], 0)
        if bottomRight[0] > frame_width: bottomRight = (frame_width, bottomRight[1])
        if bottomRight[1] > frame_height: bottomRight = (bottomRight[0], frame_height)

        roi_candidate = frame[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

        histogramScore = computeHistogramScore(ROIBGRTemplate, roi_candidate) if roi_candidate.shape[:2] == ROIBGRTemplate.shape[:2] else 0

        kp_frame, des_frame = orb.detectAndCompute(combined, None)
        matches = bf.match(des_template, des_frame) if des_frame is not None else []
        orbScore = min(len(matches) / 30.0, 1.0)

        filteredScore = 0.5 * bestVal + 0.3 * histogramScore + 0.20 * orbScore

        if currentPos is None:
            currentPos = np.array([[topLeft[0] + template_MEP.shape[1] // 2,
                                     topLeft[1] + template_MEP.shape[0] // 2]], dtype=np.float32)

        if prev_gray is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, currentPos, None)

        if status[0][0] == 1:
            flow_x, flow_y = next_pts[0]
            in_roi = (topLeft[0] <= flow_x <= bottomRight[0]) and (topLeft[1] <= flow_y <= bottomRight[1])
        else:
            in_roi = False



        if filteredScore > trackerThreshold and in_roi:

            # kilitlenme fonk
            center_x = frame_width // 2
            center_y = frame_height // 2
            half_size = 100

            roicenter_x = (topLeft[0] + bottomRight[0]) // 2
            roicenter_y = (topLeft[1] + bottomRight[1]) // 2
            trackCenterX = roicenter_x
            trackCenterY = roicenter_y

            if YOLOAndTrackerPaired:
                cv2.rectangle(frame, topLeft, bottomRight, (200, 0, 100), 2)
            else:
                cv2.rectangle(frame, topLeft, bottomRight, (0, 255, 0), 2)

            #cv2.rectangle(frame, (center_x - half_size, center_y - half_size),
            #              (center_x + half_size, center_y + half_size), (255, 0, 0), 2)

            if (center_x - half_size <= roicenter_x <= center_x + half_size) and (
                    center_y - half_size <= roicenter_y <= center_y + half_size):
                if ROIEnabled:
                    cv2.putText(frame, "eslendi!", (frame.shape[1] - 150, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ROIBlink = True
                lockedOK = True

            #cv2.putText(frame, f"Template: {bestVal:.2f}", (topLeft[0], topLeft[1] - 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #cv2.putText(frame, f"Hist: {histogramScore:.2f}", (topLeft[0], topLeft[1] - 18),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #cv2.putText(frame, f"ORB: {orbScore:.2f}", (topLeft[0], topLeft[1] - 6),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Score: {filteredScore:.2f}", (topLeft[0], topLeft[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            currentPos = np.array([[topLeft[0] + template_MEP.shape[1] // 2,
                                     topLeft[1] + template_MEP.shape[0] // 2]], dtype=np.float32)
            TrackFrameLostCounter = 0

            #cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 1)

            # ROI bölgesini al
            combined_roi = combined[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

            # ROI içinden histogram hesapla
            hsv = cv2.cvtColor(roi_candidate, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist = cv2.normalize(hist, hist).flatten()

            # Histogram görüntüsü (sol üstte sabit, küçük)
            histogramImg = np.zeros((60, 120, 3), dtype=np.uint8)
            bin_width = int(120 / len(hist))

            for i in range(len(hist)):
                x = i * bin_width
                y = int(hist[i] * 60)
                cv2.rectangle(histogramImg, (x, 60), (x + bin_width, 60 - y), (0, 255, 255), -1)

            # ORB keypoint'leri sadece ROI içinde çiz
            kp_roi, _ = orb.detectAndCompute(combined_roi, None)
            combinedROIbgr = cv2.cvtColor(combined_roi, cv2.COLOR_GRAY2BGR)
            roi_with_kp = cv2.drawKeypoints(combinedROIbgr, kp_roi, None, color=(125, 125, 125), flags=0)

            # ROI'ye keypoint'li görüntüyü yerleştir
            FilteredFrame = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            FilteredFrame[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]] = roi_with_kp

            # Histogramı frame’in sol üst köşesine yerleştir
            FilteredFrame[0:60, 0:120] = histogramImg

        else:
            TrackFrameLostCounter += 1
            if TrackFrameLostCounter > TrackFrameLostThreshold:
                trackingLost = True
                roiSelected = False
                print("Takip kaybedildi")

    if BOOSTEnabled and results is not None:
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            #print(trackingLost)
            if TrackMode == 1 and trackingLost == False:
                if topLeft[0] >= x1 and x2 >= bottomRight[0] and topLeft[1] >= y1 and y2 >= bottomRight[1]:
                    #print("track roi yolonun roi icinde")
                    YOLOAndTrackerPaired = True
                else:
                    YOLOAndTrackerPaired = False
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 200), 2)
                    label = model.names[int(cls)]
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            else:
                YOLOAndTrackerPaired = False
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 200), 2)
                label = model.names[int(cls)]
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    #UI yazdirma kismi
    if HUDEnabled:
        if ROIEnabled:
            if ROIBlink:
                cv2.rectangle(frame, (fixedROIX, fixedROIY),
                              (fixedROIX + fixedROISize, fixedROIY + fixedROISize), (255, 0, 0), 2)
            if not lockedOK:
                if LineEnabled and trackingLost is not True:
                    cv2.line(frame, (int(roicenter_x), int(roicenter_y)), (frame.shape[1] // 2, frame.shape[0] // 2),
                             (255, 0, 0), 1, cv2.LINE_AA)


        cv2.putText(frame, "DEMO Version", (frame.shape[1]-160, 25), font, generalFontScale, (200, 0, 0), 1,cv2.LINE_AA)
        if BOOSTEnabled and TrackMode == 0:
            cv2.putText(frame, "Secilen Mod: Takip + AI Tespit", (15, 25), font, generalFontScale, (200, 0, 0), 1,cv2.LINE_AA)
        elif TrackMode == 0:
            cv2.putText(frame, "Secilen Mod: Takip", (15, 25), font, generalFontScale, (200, 0, 0), 1,cv2.LINE_AA)
        elif TrackMode == 1:
            if BOOSTEnabled:
                cv2.putText(frame, "Secilen Mod: AI Destekli Takip", (15, 25), font, generalFontScale, (200, 0, 0), 1,cv2.LINE_AA)
            else:
                cv2.putText(frame, "Secilen Mod: AI Destekli Takip (BOOST aciniz !!)", (15, 25), font, generalFontScale, (200, 0, 0), 1,cv2.LINE_AA)
        if trackingLost is False:
            cv2.putText(frame, "Takipte", (15, 60), font, generalFontScale, (50, 200, 50), 1,cv2.LINE_AA)
        else:
            cv2.putText(frame, "Takip Disi", (15, 60), font, generalFontScale, (50, 50, 200), 1,cv2.LINE_AA)
        cv2.putText(frame, "Tracker Demonstration Program", (15, frame.shape[0]-25), font, generalFontScale, (200, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(frame, "(x,y)", (80, frame.shape[0]-210), font, generalFontScale, (50, 150, 50), 1,cv2.LINE_AA)
        cv2.putText(frame, "(" + str(trackCenterX) + ","+str(trackCenterY)+ ")", (48, frame.shape[0]-120), font, generalFontScale, (50, 150, 50), 1,cv2.LINE_AA)
        cv2.rectangle(frame,(15,frame.shape[0]-50),(190,frame.shape[0]-200),(50,150,50),1)

        #background color olabilir ama kotu duruyor
        #cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

        if MetricsEnabled:
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 1)


    if FilteredFrame is not None and not trackingLost:
        # kırpma islemi roinin etrafinda cerceve yapiyor ve ekranin ufak bir kismini sunuyor
        if currentPos is not None:
            # ROI center koord
            roi_cx, roi_cy = int(currentPos[0][0]), int(currentPos[0][1])

            # 480x360 crop için yarı genişlik/yükseklik
            crop_w, crop_h = 240, 180

            # Crop bölge sınırlari belirle
            x1 = max(roi_cx - crop_w, 0)
            y1 = max(roi_cy - crop_h, 0)
            x2 = min(roi_cx + crop_w, frame.shape[1])
            y2 = min(roi_cy + crop_h, frame.shape[0])

            crop = FilteredFrame[y1:y2, x1:x2].copy()

            # Crop yeniden boyutlandırma
            crop = cv2.resize(crop, (480, 360))
            if histogramImg is not None:
                crop[0:60, 0:120] = histogramImg

            cv2.imshow("Filter View", crop)
        #cv2.imshow("Filter", FilteredFrame)
        #cv2.imshow("Template Fusion", template_MEP)

    cv2.imshow("Frame", frame)
    cv2.imshow("Kontrol Paneli", panel)
    selectedButton = 0
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #prevTime = time.time()

    if key == 27 or exitFlag:
        break

cap.release()
cv2.destroyAllWindows()

