from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from PyQt5.uic import loadUi

from PIL import Image
import tools, cv2, time, act, os
import matplotlib.pyplot as plt

class Widget(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("assets/ASI.ui",self)
        self.setWindowTitle("ASI")

        self.openImg.clicked.connect(self.addimageToview)
        self.conversionChoice.currentIndexChanged.connect(self.Conversion)
        self.histChoice.currentIndexChanged.connect(self.Histograms)
        self.SaveImg.clicked.connect(self.save_Image)
        self.RefreshImg.clicked.connect(self.showimage)
        self.startTrans.clicked.connect(self.Transformations)
        self.transChoice.currentIndexChanged.connect(self.checkTransOption)
        self.contourChoice.currentIndexChanged.connect(self.ContourDetection)
        self.smooChoice.currentIndexChanged.connect(self.Smoothing)
        self.ThresholdChoice.currentIndexChanged.connect(self.checkTransOption)
        self.startThreshold.clicked.connect(self.Threshold)
        self.houghChoice.currentIndexChanged.connect(self.checkTransOption)
        self.startHough.clicked.connect(self.Hough)

    def addimageToview(self):
        try:
            imageFile = QFileDialog.getOpenFileName(None, "Open image", os.getcwd()+"/Images", "Image Files (*.png *.jpg *.bmp *.jpeg *.png *.jfif)")
            self.imagename, self.imagenameCV = str(imageFile[0]), str(imageFile[0])
            self.showimage()
            self.pixel = tools.getImage(self.imagename)
            self.pixelCV = cv2.imread(self.imagenameCV)
        except: pass

    def showimage(self):
        self.codeImg.setPixmap(QtGui.QPixmap(self.imagename))
        self.CVImg.setPixmap(QtGui.QPixmap(self.imagenameCV))

    def showimageResult(self, imgNameCode, imgNameCV):
        self.codeImg.setPixmap(QtGui.QPixmap(imgNameCode))
        self.CVImg.setPixmap(QtGui.QPixmap(imgNameCV))

    def save_Image(self):
        try:
            self.imagename = self.savedImg
            self.imagenameCV = self.savedCVImg
            self.pixel = tools.getImage(self.imagename)
            self.pixelCV = cv2.imread(self.imagenameCV)
        except: pass
    
    def Conversion(self):
            Value = self.conversionChoice.currentIndex()
            if Value == 0: self.showimage()
            else:
                start_time = time.time()
                newImg, name, newImgCV = act.ImageConversion(self.pixel, self.pixelCV, Value)
                print("--- %s seconds ---" % (time.time() - start_time))

                self.savedImg, self.savedCVImg = tools.SaveImage(newImg, newImgCV, name)
                self.showimageResult('Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png')

    def Histograms(self):
            Value = self.histChoice.currentIndex()
            if Value == 0: self.showimage()
            elif Value == 4:
                newImg = act.equalization_Hist(self.pixel)
                newImgCV = cv2.equalizeHist(cv2.cvtColor(self.pixelCV, cv2.COLOR_BGR2GRAY))

                Image.fromarray(newImg.astype('uint8')).save('Images/Saved/Equalized.png', format='png')
                cv2.imwrite('Images/SavedCV/Equalized.png', newImgCV)
                self.showimageResult('Images/Saved/Equalized.png', 'Images/SavedCV/Equalized.png')

                self.savedImg = 'Images/Saved/Equalized.png'
                self.savedCVImg = 'Images/SavedCV/Equalized.png'
            else:
                hist, histCV = act.histTreatments(self.pixel, self.pixelCV, Value)

                fig = plt.figure(figsize=(15, 8))
                ax1 = fig.add_subplot(121)
                plt.title("Code")
                ax2 = fig.add_subplot(122)
                plt.title("OpenCV")
                if len(hist) == 3:
                    ax1.plot(hist[0], color = 'red'), ax1.plot(hist[1], color = 'green'), ax1.plot(hist[2], color = 'blue')
                    ax2.plot(histCV[2], color = 'red'), ax2.plot(histCV[1], color = 'green'), ax2.plot(histCV[0], color = 'blue')
                else:
                    ax1.plot(hist)
                    ax2.plot(histCV)
                plt.show()

    def checkTransOption(self):
        ValueTran = self.transChoice.currentIndex()
        if ValueTran == 0:
            self.transX.setEnabled(False), self.transX.setText('')
            self.transY.setEnabled(False), self.transY.setText('')
            self.transRotation.setEnabled(False), self.transRotation.setText('')
            self.transScale.setEnabled(False), self.transScale.setText('')
        elif ValueTran == 1 or ValueTran == 4:
            self.transX.setEnabled(True), self.transX.setText('')
            self.transY.setEnabled(True), self.transY.setText('')
            self.transRotation.setEnabled(False), self.transRotation.setText('')
            self.transScale.setEnabled(False), self.transScale.setText('')
        elif ValueTran == 2:
            self.transX.setEnabled(False), self.transX.setText('')
            self.transY.setEnabled(False), self.transY.setText('')
            self.transRotation.setEnabled(True), self.transRotation.setText('')
            self.transScale.setEnabled(False), self.transScale.setText('')
        elif ValueTran == 3:
            self.transX.setEnabled(False), self.transX.setText('')
            self.transY.setEnabled(False), self.transY.setText('')
            self.transRotation.setEnabled(False), self.transRotation.setText('')
            self.transScale.setEnabled(True), self.transScale.setText('')

        if self.ThresholdChoice.currentIndex() == 0:
            self.threMin.setEnabled(False), self.threMin.setText('')
            self.threMax.setEnabled(False), self.threMax.setText('')
        else:
            self.threMin.setEnabled(True), self.threMin.setText('')
            self.threMax.setEnabled(True), self.threMax.setText('')

        if self.houghChoice.currentIndex() == 0:
            self.threHough.setEnabled(False), self.threHough.setText('')
            self.MinRadiusHough.setEnabled(False), self.MinRadiusHough.setText('')
            self.MaxRadiusHough.setEnabled(False), self.MaxRadiusHough.setText('')
        elif self.houghChoice.currentIndex() == 1:
            self.threHough.setEnabled(True), self.threHough.setText('200')
            self.MinRadiusHough.setEnabled(False), self.MinRadiusHough.setText('')
            self.MaxRadiusHough.setEnabled(False), self.MaxRadiusHough.setText('')
        elif self.houghChoice.currentIndex() == 2:
            self.threHough.setEnabled(True), self.threHough.setText('150')
            self.MinRadiusHough.setEnabled(True), self.MinRadiusHough.setText('40')
            self.MaxRadiusHough.setEnabled(True), self.MaxRadiusHough.setText('120')
        
    def Transformations(self):
            ValueTrans = self.transChoice.currentIndex()
            if ValueTrans == 0: self.showimage()
            else:
                x = tools.checkNull(self.transX.text())
                y = tools.checkNull(self.transY.text())
                angle = tools.checkNull(self.transRotation.text())
                factor = tools.checkNull(self.transScale.text())
                newImg, newImgCV, name = act.TransformationsGeo(self.pixel, self.pixelCV, x, y, int(angle), int(factor), ValueTrans)

                self.savedImg, self.savedCVImg = tools.SaveImage(newImg, newImgCV, name)
                self.showimageResult('Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png')

    def ContourDetection(self):
            Value = self.contourChoice.currentIndex()
            if Value == 0: self.showimage()
            else:
                newImg, newImgCV, name = act.ContourDetection(self.pixel, self.pixelCV, Value)

                self.savedImg, self.savedCVImg = tools.SaveImage(newImg, newImgCV, name)
                self.showimageResult('Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png')

    def Smoothing(self):
        Value = self.smooChoice.currentIndex()
        if Value == 0: self.showimage()
        else:
            newImg, newImgCV, name = act.Smoothing(self.pixel, self.pixelCV, Value)

            self.savedImg, self.savedCVImg = tools.SaveImage(newImg, newImgCV, name)
            self.showimageResult('Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png')

    def Threshold(self):
        Value = self.ThresholdChoice.currentIndex()
        if Value == 0: self.showimage()
        else:
            mn = tools.checkNull(self.threMin.text())
            mx = tools.checkNull(self.threMax.text())
            newImg, newImgCV, name = act.Threshold(self.pixel, self.pixelCV, Value, int(mn), int(mx)) 

            self.savedImg, self.savedCVImg = tools.SaveImage(newImg, newImgCV, name) 
            self.showimageResult('Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png')  

    def Hough(self):
        Value = self.houghChoice.currentIndex()
        if Value == 0: self.showimage()
        else:
            threHough = tools.checkNull(self.threHough.text())
            minHough = tools.checkNull(self.MinRadiusHough.text())
            maxHough = tools.checkNull(self.MaxRadiusHough.text())
            
            start_time = time.time()
            newImg, newImgCV, name = act.Hough(self.pixel, self.pixelCV, Value, int(threHough), int(minHough), int(maxHough))  
            print("--- %s seconds ---" % (time.time() - start_time))
            
            self.savedImg, self.savedCVImg = tools.SaveImage(newImg, newImgCV, name) 
            self.showimageResult('Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png')    

if __name__ == '__main__':
    app = QApplication([])
    window = Widget()
    window.show()
    app.exec_()