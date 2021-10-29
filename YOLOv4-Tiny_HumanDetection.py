import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights","yolov4-tiny-custom.cfg")

classes = ["Human"]

cap = cv2.VideoCapture("12.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = 1000/fps
result = cv2.VideoWriter('Video-demo.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    # [blobFromImage] tạo đốm màu 4 chiều từ hình ảnh. Tùy chọn thay đổi kích thước và cắt hình ảnh từ trung tâm, trừ giá trị trung bình,
    # chia tỷ lệ giá trị theo tỷ lệ, hoán đổi kênh Xanh lam và Đỏ.
    blob = cv2.dnn.blobFromImage(frame, 1./255, (416,416),(0,0,0),swapRB=True, crop=False)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    # cv2.waitKey(0)
    net.setInput(blob)


    output_layers_names = net.getUnconnectedOutLayersNames()          #['yolo_30', 'yolo_37']

    layerOutputs = net.forward(output_layers_names)                  #2 Ma trận kích thước (507,7) và (2028,7)

    boxes = []
    confidences = []
    class_ids = []

    # Show information on the screen
    for output in layerOutputs:     #có 2 output là kích thước (507,7) và (2028,7), mỗi hàng là 1 bouding box
        for detection in output:  # (duyệt từng hàng)  detection: mảng gồm 7 giá trị: detection[0] và [1]: tọa độ tâm bouding box chuẩn hóa (0<x,y <1)
                                                                                        #detection[2] và [3]: chiều rộng và chiều cao của bouding box chuẩn hóa (0<w,h<1)
            scores = detection[5:7]  # Chứa confidence của các class trong code này có 2 class
            class_id = np.argmax(scores)      #Chỉ số của class mà có confidence lớn hơn
            confidence = scores[class_id]  # confidence: độ tin cậy của boundingbox, nhận giá trị từ 0 đến 1
            # print(confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * width)        #Tọa độ tâm của bounding box thực tế trên ảnh
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)               #Chiều rộng, chiều cao thực tế của bounding box so với kích thước ảnh
                h = int(detection[3] * height)

                x = int(center_x - w / 2)                #Tọa độ điểm góc trên bên trái của bounding box
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])  # x,y,w,h: tọa độ góc điểm góc trên, bên trái của boundingbox

                confidences.append((float(confidence)))         #danh sách các confidence cảu các bouding box
                class_ids.append(class_id)                      #Danh sách chỉ số của class có confidence lớn hơn (nhận giá trị 0 hoặc 1)

    # print(boxes)
    # print(confidences)
    # print(class_ids)

    # Chọn ra những box có độ tin cậy cao, đại diện box là chỉ số của class
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)            #0.5: Ngưỡng lọc confidence tương ứng với box, để loại box
    # print("Indexes: ",indexes)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    colors = [(0, 0, 255), (0, 255, 0)]


    for i in range(len(boxes)):          #len(boxes) = 2
         if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(frame, label + " " + confidence, (x, y - 5), font, 1, color, 2)


    cv2.imshow('Video', frame)
    result.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
result.release()
cv2.destroyAllWindows()

