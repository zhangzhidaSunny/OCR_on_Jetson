
from PyQt5.QtWidgets import QWidget,QTextEdit,QComboBox, QLabel,QPushButton,QHBoxLayout,QVBoxLayout,QLineEdit,QFileDialog,QApplication
from PyQt5.QtGui import QPixmap
import sys
import cv2
import os
import fastdeploy as fd

class OCRApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        okButton = QPushButton("OCR 识别")
        selButton = QPushButton("选择图片")
        self.te1 = QTextEdit()

        lbl2 = QLabel('识别结果', self)
        pixmap = QPixmap()
        pixmap1 = QPixmap()
        hbox2 = QHBoxLayout()
        hbox2.addWidget(lbl2)
        hbox2.addWidget(self.te1)
        self.cb = QComboBox()
        self.cb.addItem('CPU')
        self.cb.addItem('GPU')
        hbox = QHBoxLayout()
        hbox.addWidget(selButton)
        hbox.addWidget(okButton)
        hbox.addWidget(self.cb)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox2)

        vbox.addLayout(hbox)

        self.lbl = QLabel()
        self.lbl.setPixmap(pixmap)
        self.lbl1 = QLabel()
        self.lbl1.setPixmap(pixmap1)
        hWholeBox = QHBoxLayout()
        hWholeBox.addWidget(self.lbl)
        hWholeBox.addWidget(self.lbl1)
        vWholeBox= QVBoxLayout()
        vWholeBox.addLayout(hWholeBox)
        vWholeBox.addLayout(vbox)

        self.setLayout(vWholeBox)
        self.setGeometry(500, 500, 800, 400)
        self.setWindowTitle('OCR 识别程序')
        self.show()
        selButton.clicked.connect(self.showDialog)
        okButton.clicked.connect(self.processOCR)

    def cv_show(self, name, img):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './pic')
        print(fname[0])
        self.path = fname[0]
        try:
            img = cv2.imread(fname[0])
            x, y = img.shape[0:2]
            resize_img=cv2.resize(img,(640,int(640*x/y)))
            border=int((x-int(640*x/y))/2)
            print(border)
            cImg=cv2.copyMakeBorder(resize_img,int((640-int(640*x/y))/2),int((640-int(640*x/y))/2),0,0,cv2.BORDER_CONSTANT,value=0)

            # self.cv_show("resize",cImg)
            cv2.imwrite(fname[0], cImg)
            pPic = QPixmap(fname[0])
            height = pPic.height()
            width = pPic.width()
            pPic = pPic.scaled(width/2, height/2)
            self.lbl.setPixmap(pPic)

        except Exception:
            print (Exception.args);





    def processOCR(self):
        try:

            det_model = "ch_PP-OCRv3_det_infer"
            cls_model = "ch_ppocr_mobile_v2.0_cls_infer"
            rec_model = "ch_PP-OCRv3_rec_infer"
            rec_label_file = "ppocr_keys_v1.txt"
            image = ""
            cls_bs = 1
            rec_bs = 6

            device = "cpu"
            cpu_thread_num = 9
            device_id = 0
            det_option = fd.RuntimeOption()
            cls_option = fd.RuntimeOption()
            rec_option = fd.RuntimeOption()

            # det_option.set_cpu_thread_num(cpu_thread_num)
            # cls_option.set_cpu_thread_num(cpu_thread_num)
            # rec_option.set_cpu_thread_num(cpu_thread_num)
            if self.cb.currentText().lower() =="gpu":
                det_option.use_gpu(device_id)
                cls_option.use_gpu(device_id)
                rec_option.use_gpu(device_id)
                det_option.use_trt_backend()
                cls_option.use_trt_backend()
                rec_option.use_trt_backend()

                # 设置trt input shape
                # 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
                det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                               [1, 3, 960, 960])
                cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                               [cls_bs, 3, 48, 320],
                                               [cls_bs, 3, 48, 1024])
                rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                               [rec_bs, 3, 48, 320],
                                               [rec_bs, 3, 48, 2304])

                # 用户可以把TRT引擎文件保存至本地
                det_option.set_trt_cache_file(det_model + "/det_trt_cache.trt")
                cls_option.set_trt_cache_file(cls_model + "/cls_trt_cache.trt")
                rec_option.set_trt_cache_file(rec_model + "/rec_trt_cache.trt")


            # Detection模型, 检测文字框
            det_model_file = os.path.join(det_model, "inference.pdmodel")
            det_params_file = os.path.join(det_model, "inference.pdiparams")
            # Classification模型，方向分类，可选
            cls_model_file = os.path.join(cls_model, "inference.pdmodel")
            cls_params_file = os.path.join(cls_model, "inference.pdiparams")
            # Recognition模型，文字识别模型
            rec_model_file = os.path.join(rec_model, "inference.pdmodel")
            rec_params_file = os.path.join(rec_model, "inference.pdiparams")
            rec_label_file = rec_label_file


            det_model = fd.vision.ocr.DBDetector(
                det_model_file, det_params_file, runtime_option=det_option)

            cls_model = fd.vision.ocr.Classifier(
                cls_model_file, cls_params_file, runtime_option=cls_option)

            rec_model = fd.vision.ocr.Recognizer(
                rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

            # 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
            ppocr_v3 = fd.vision.ocr.PPOCRv3(
                det_model=det_model, cls_model=None, rec_model=rec_model)

            # 给cls和rec模型设置推理时的batch size
            # 此值能为-1, 和1到正无穷
            # 当此值为-1时, cls和rec模型的batch size将默认和det模型检测出的框的数量相同

            ppocr_v3.cls_batch_size = cls_bs
            ppocr_v3.rec_batch_size = rec_bs
            # 预测图片准备
            im = cv2.imread(self.path)
            e1 = cv2.getTickCount()
            # 预测并打印结果
            result = ppocr_v3.predict(im)
            e2 = cv2.getTickCount()
            time = (e2 - e1) / cv2.getTickFrequency()

            print(result)
            self.te1.append(str(result))
            self.te1.append("检测耗时"+str(time)+"秒")
            # 可视化结果
            vis_im = fd.vision.vis_ppocr(im, result)
            cv2.imwrite("visualized_result.jpg", vis_im)
            pPic1 = QPixmap("visualized_result.jpg")
            height = pPic1.height()
            width = pPic1.width()
            pPic1 = pPic1.scaled(width / 2, height / 2)
            self.lbl1.setPixmap(pPic1)
            print("Visualized result save in ./visualized_result.jpg")
            self.repaint()
            print("time=")
            print(time)

        except Exception as e:
            self.te1.append(str(e))
            print(e);


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = OCRApp()

    sys.exit(app.exec_())