import cv2
from queue import Queue
from yolov5_detect_image import Y5Detect, draw_boxes_tracking, draw_det_when_track
import time
from kthread import KThread
import numpy as np
from mot_sort.mot_sort_tracker import Sort
from mot_sort import untils_track


y5_model = Y5Detect(weights="../new_core/yolov5_detect/model_head/y5headbody_v2.pt")
class_names = y5_model.class_names
mot_tracker = Sort(class_names)


class InfoCam(object):
    def __init__(self, cam_name):
        self.cap = cv2.VideoCapture(cam_name)


def video_capture(cam, frame_detect_queue, frame_origin_queue):
    frame_count = 0

    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    while cam.cap.isOpened():
        ret, frame_ori = cam.cap.read()
        # time.sleep(0.01)
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)
        frame_detect_queue.put(image_rgb)
        frame_origin_queue.put([frame_ori, frame_count])
        print("frame_count: ", frame_count)
        frame_count += 1

    cam.cap.release()


def inference(cam, frame_detect_queue, detections_queue): #, tracking_queue):
    while cam.cap.isOpened():
        image_rgb = frame_detect_queue.get()
        boxes, labels, scores, detections_sort = y5_model.predict_sort(image_rgb, label_select=["head"])
        # for i in range(len(scores)):
        #     detections_tracking = bboxes[i].append(scores[i])
        detections_queue.put([boxes, labels, scores, image_rgb, detections_sort])
        # tracking_queue.put([detections_tracking])

    cam.cap.release()


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def to_tlwh(tlbr):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = tlbr.copy()
    # ret[2:] += ret[:2]
    box = []
    for bbox in ret:
        w = int(bbox[2]) - int(bbox[0])
        h = int(bbox[3]) - int(bbox[1])
        box.append([int(bbox[0]) + w/2, int(bbox[1]) + h/2, w, h])
    return box


def tracking(cam, frame_origin_queue, detections_queue, tracking_queue):
    """
    :param cam:
    :param frame_origin_queue:
    :param detections_queue:
    :param tracking_queue:
    :return:
    Tracking using SORT. Hungary + Kalman Filter.
    Using mot_tracker.update()
    Input: detections [[x1,y1,x2,y2,score,label],[x1,y1,x2,y2,score, label],...], use np.empty((0, 5)) for frames without detections
    Output: [[x1,y1,x2,y2,id1, label],[x1,y1,x2,y2,id2, label],...]
    """
    region_track = np.array([[0, 0],
                             [2560, 0],
                             [2560, 1440],
                             [0, 1440]])

    while cam.cap.isOpened():
        boxes, labels, scores, image_rgb, detections_sort = detections_queue.get()
        if len(boxes) == 0:
            detections = np.empty((0, 5))
        else:
            detections = detections_sort
            # check and select the detection is inside region tracking
            detections = untils_track.select_bbox_inside_polygon(detections, region_track)

        track_bbs_ids, unm_trk_ext = mot_tracker.update(detections, image=image_rgb)
        # print("labels, scores", labels, scores)
        # print(track_bbs_ids)
        tracking_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext])

    cam.cap.release()


def drawing(cam, tracking_queue, frame_origin_queue, frame_final_queue, show_det=True):
    while cam.cap.isOpened():
        frame_origin, frame_count = frame_origin_queue.get()
        track_bbs_ids, boxes, labels, scores, unm_trk_ext = tracking_queue.get()
        if frame_origin is not None:
            image = draw_boxes_tracking(frame_origin, track_bbs_ids, scores=scores, labels=labels,
                                        class_names=class_names, track_bbs_ext=unm_trk_ext)
            if show_det:
                image = draw_det_when_track(frame_origin, boxes, scores=scores, labels=labels,
                                            class_names=class_names)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if frame_final_queue.full() is False:
                frame_final_queue.put([image, frame_count])
            else:
                time.sleep(0.001)
    cam.cap.release()


def save_debug_image(frame_count, image):
    cv2.imwrite("/home/vuong/Desktop/Project/GG_Project/green-clover-montessori/new_core/debug_image/test_" + str(
        frame_count) + ".png", image)


def main():
    frame_detect_queue = Queue(maxsize=1)
    frame_origin_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    tracking_queue = Queue(maxsize=1)
    frame_final_queue = Queue(maxsize=1)
    input_path = "/home/vuong/Downloads/output-1617592346.8862946_part3.mp4"
    cam = InfoCam(input_path)

    thread1 = KThread(target=video_capture, args=(cam, frame_detect_queue, frame_origin_queue))
    thread2 = KThread(target=inference, args=(cam, frame_detect_queue, detections_queue))
    thread3 = KThread(target=tracking, args=(cam, frame_origin_queue, detections_queue, tracking_queue))
    thread4 = KThread(target=drawing, args=(cam, tracking_queue, frame_origin_queue, frame_final_queue))

    thread_manager = []
    thread1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    thread1.start()
    thread_manager.append(thread1)
    thread2.daemon = True
    thread2.start()
    thread_manager.append(thread2)
    thread3.daemon = True
    thread3.start()
    thread_manager.append(thread3)
    thread4.daemon = True
    thread4.start()
    thread_manager.append(thread4)

    while cam.cap.isOpened():
        cv2.namedWindow('output')
        image, frame_count = frame_final_queue.get()
        image = cv2.resize(image, (1400, 640))
        cv2.imshow('output', image)
        if frame_count >= 1550 and (frame_count <= 1800):
            KThread(target=save_debug_image, args=(frame_count, image)).start()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow('output')
            break

    for t in thread_manager:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

