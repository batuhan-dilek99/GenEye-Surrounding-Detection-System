# yolov5-android-tflite

Bu çalışmamızda yolov5 ile özel verilerde eğitim yaptık ve eğitim sonucunda oluşan ağırlıkları kullanrak tensorflow lite apisinde (android) kullanıcaz. 

pip install -r requirements.txt

pip install tensorflow==2.3.1

python export.py --weights last.pt --include tflite --img 320
