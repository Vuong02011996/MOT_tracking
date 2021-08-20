# Phân tích vấn đề.

+ Object tracking là bài toán theo dõi một hoặc nhiều đối tượng chuyển động theo thời gian trong một video.
+ Tracking ngoài việc xác định bounding box còn phải quan tâm đến các vấn đề sau:
    + ID của mỗi đối tượng cần đảm bảo không đổi qua các frame.
    + Khi đối tượng bị che khuất hay biến mất sau một vài frame , hệ thống cần đảm bảo nhận diện lại đúng ID khi đối tượng xuất hiện lại.
    + Các vấn đề liên quan đến tốc độ xử lí realtime, và tính ứng dụng cao.
    + ...
    
# Phân loại 

+ Single Object Tracking (SOT)
+ Multiple Object Tracking (MOT)

# Dataset và Metric đánh giá.

+ **Dataset** 
    + MOT Challenge 
    + ImageNet VID
    
+ **Metric đánh giá** 
    + FP: Tổng số lần đối tượng được phát hiện mặt dù thực tế không có.
    + FN: Tổng số lần đối tượng có mà không được phát hiện.
    + Recall: tổng số lượng đối tượng phát hiện
    + ID Switches: Tổng số 1 lần đối tượng bị gán cho ID mới trong suốt quá trình tracking video.
    + MOTA. Mutiple Object Tracking Accuracy
      + ![](MOTA.png)
    + MOTP.  Mutiple Object Tracking Precision
      + ![](MOTP.png)
    + MT(Most Tracked Target): tính từ 80% video -> hết.[link](https://github.com/cheind/py-motmetrics)
    + PT(partially_tracked): Tính từ 20% - 80% video.
    + ML(Most Lost Target): Tính từ đầu đến 20% video.
    + Frag: tổng số lần đối tượng bị phân mảnh.
    + Hz: tốc độ tracking.
    + Công thức :
      + ![](metrics.png)

+ **Cách đánh giá**
  + Đánh giá dựa trên GroundTruth data. But only some data have GT label.MOT15, MOT16, MOT17, MOT20, ..almost is person.
  + ***Bảng xếp hạng các thuật toán tracking cập nhật thường niên.***
    + [Link](https://motchallenge.net/results/MOT17/)
  + Code evaluation
    + [MOT challenge](https://github.com/dendorferpatrick/MOTChallengeEvalKit)
    + [HOTA metrics](https://github.com/JonathonLuiten/TrackEval)
    + [metrics](https://github.com/luanshiyinyang/awesome-multiple-object-tracking#metrics)
    + [py-metrics](https://github.com/cheind/py-motmetrics)
    + [Format](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official)
# Các vấn đề cần quan tâm nhất trong Object Tracking.

+ Multiple Object Tracking 
    + Phát hiện "tất cả" các đối tượng: ```đảm bảo tính chính xác của quá trình detect là vô cùng quan trọng.```
    + Đối tượng bị che khuất 1 phần hoặc toàn bộ : ```chỉ dựa vào objection là không đủ để giải quyết vấn đề.```
    + Đối tượng ra khỏi phạm vi khung hình sau đó xuất hiện lại: 
      ```cần giải quyết tốt vấn đề nhận dạng lại đối tượng kể cả việc che khuất hay biến mất để giảm số lượng ID switches xuống mức thấp nhất có thể.```
    + Các đối tượng có quỹ đạo chuyển động giao nhau hoặc chồng chéo lên nhau:```có thể gán nhầm ID```
  
+ Realtime Object Tracking 
    + Realtime : đảm bảo tốc độ đưa ra output nhanh hơn hoặc bằng tốc độ input đưa vào.
    + Có thể bỏ qua một vài frame không xử lí cho đến khi frame hiện tại được xử lí xong.
      
# Giới thiệu thuật toán SORT (Simple Online Real Time Object Tracking).

* Khung quá trình xử lí SORT như sau:
    1. **Detect**: phát hiện vị trí các đối tượng trong frame (Object detection)
    2. **Predict**: Dự đoán vị trí mới của đối tượng dựa vào các frame trước đó.
    3. **Associate(Update)**: Liên kết vị trí detected với các vị trí dự đoán được để gán ID tương ứng.
  

### 1. Giải thuật Hungary 
+ *Bài Toán*: Có n detection(i = 1, 2, .., n) và n track predicted (j = 1, 2, .., n), cần liên kết mỗi dectection với mỗi track tương 
ứng sao cho sai số của việc liên kết là nhỏ nhât. *Sai số* được tính là khoảng cách trong không gian vector(bbox vector) của i và j.

+ Ref Hungary in [here](https://www.youtube.com/watch?v=cQ5MsiGaDY8)

+ Thuật toán : ```Biến đổi ma trận (cộng trừ vào các hàng hoặc cột) để đưa về ma trận có n phần tử bằng 0 nằm ở các hàng và cột khác nhau 
sau đó lấy ra phương án tối ưu là vị trí chứa các phần tử 0 này ```
  1. Bước 1: Trừ mỗi hàng của ma trận C cho phần tử nhỏ nhất của hàng đó. Tiếp theo cũng ma trận sau khi trừ tiếp tục trừ mỗi cột cho phần tử nhỏ 
  nhất của cột đó. Ta được ma trận C' mỗi hàng mỗi cột có ít nhất một phần tử bằng 0.
  2. Bước 2: Vẽ một số *tối thiểu* các đường thằng qua các dòng và cột mà đảm bảo mọi phần tử 0 được đi qua. 
  3. Bước 3 : Nếu có n đường thẳng được vẽ kết thúc thuật toán và tiến hành phân công công việc. Nếu số đường thẳng nhỏ hơn n vẫn chưa tìm được phương án tối ưu .Qua bước tiếp theo.
  4. Bước 4: Mỗi hàng hoặc cột có đường thẳng đi qua ta gọi là các hàng hoặc cột thiết yếu. Các hàng cột còn lại là không thiết yếu. Tìm phần tử nhỏ nhất không nằm trong các hàng cột không thiết yếu.
  Tiến hành trừ *mỗi hàng không thiết yếu* cho phần tử nhỏ nhất ấy và *cộng phần tử nhỏ nhất ấy cho cột thiết yếu*. Quay lại bước 2.
     
  ```   
   Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> from scipy.optimize import linear_sum_assignment
    >>> row_ind, col_ind = linear_sum_assignment(cost)
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind].sum()
    5
    ```

### 2. Bộ lọc Kalman(kalman Filter)

+ Trong object tracking Kalman Filter được biết đến với vai trò dự đoán các trạng thái của đối tượng hiện tại dựa vào các tracking trong quá khứ và update lại các detection sau khi liên kết với các track trước đó.

+ **Dự đoán(Predict)**: để dự đoán trạng thái của quá trình ngẫu nhiên ta dự đoán các mean và covariance.
+ **Hiệu chỉnh(Update)**: Áp dụng Bayes. 

1. Lý thuyết(The Math to understand)
   1. Mean and Expectation.
   2. Variance and Standard deviation
   3. Matrix operations.(Dot product, add, transform, inverter, ...)
   4. Covariance and Covariance matrix
   5. Likehood, Maximum Likehood Estimate(MLE).
   6. Eigenvector and eigenvalue.(covariance matrix-COV)
   7. Normal Distribution(Gauss) - PDF (Ham mat do xac suat) - 69.3; 95.7; 97.5.
   8. Measurement, estimate, update, accuracy, predict, precision, ...
   9. Error(uncertainly): measurement error(r), estimate error(p), 
   10. FIVE Kalman Filter Equation
2. Kalman Filter tutorial 
   1. [https://www.kalmanfilter.net/kalman1d.html](https://www.kalmanfilter.net/kalman1d.html)
   2. [https://codelungtung.wordpress.com/2019/05/29/bo-loc-kalman/](https://codelungtung.wordpress.com/2019/05/29/bo-loc-kalman/)
  
### Sort explain 

1. Ma trận hiệp phương sai. [link](https://minhng.info/toan-hoc/ma-tran-hiep-phuong-sai-covariance-matrix.html)

2. Flow code:
   1. Input: [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
   2. Output: [[x1,y1,x2,y2,id1],[x1,y1,x2,y2,id2],...]

Mô hình :
![Mô hình SORT ](mo_hinh_sort.png)

# Some good resource for tracking
## Comparing
* FairMOT, DeepSort, JDE using GPU and FPS not good.And training model reID with persons.
* SDE - Separate Detection and Embedding: DEEPSORT
* JDE (Joint Detection and Embedding): JDE and FairMOT
* 
+ [Danh gia](https://github.com/luanshiyinyang/awesome-multiple-object-tracking)
+ [Compare: deepsort, fairmot, jde](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot)
+ [DeepSort](https://github.com/nwojke/deep_sort)
+ [FairMOT - best MOTA](https://github.com/ifzhang/FairMOT/tree/cb56feaee38e697db5a8f051c39a61edd6e09f9f)
+ [JDETracker-Towards Real-Time Multi-Object Tracking](https://github.com/Zhongdao/Towards-Realtime-MOT)

## Good source
1. SORT_OH 
   1. [source](https://github.com/mhnasseri/sort_oh/blob/main/libs/tracker.py)
   2. MOT 
      - [MOT16](https://motchallenge.net/method/MOT=4046&chl=5) 
      - [MOT17](https://motchallenge.net/method/MOT=4046&chl=10)
   3. Paper [https://arxiv.org/pdf/2103.04147.pdf](https://arxiv.org/pdf/2103.04147.pdf)
2. FairMOT 
   1. [MOT17](https://motchallenge.net/method/MOT=3015&chl=10)
3. norfair: custom python library. [link](https://github.com/tryolabs/norfair)


# How to submission to MOT challenge 
    
[link](https://motchallenge.net/instructions/)
[code to create file zip to submiss](https://github.com/mhnasseri/sort_oh/blob/main/tracker_app.py)



  

