import numpy as np
from tqdm import tqdm
import time
from glob import glob
import os
from new_core.mot_sort.mot_sort_tracker import Sort

if __name__ == '__main__':
    path_data_submit = "/home/vuong/Downloads/MOT17/test/"
    outFolder = "/home/vuong/Desktop/Project/GG_Project/green-clover-montessori/new_core/mot_sort/data_submit/"
    mot_tracker = Sort()

    list_name_folder = glob(path_data_submit + "*/")
    # train sequences
    total_time = 0.0
    total_frames = 0
    for seq in list_name_folder:
        common_path = seq
        seq_dets = np.loadtxt('%s/det/det.txt' % common_path, delimiter=',')

        with open('%s/%s.txt' % (outFolder, seq.split("/")[-2]), 'w') as out_file:
            print("\nProcessing %s." % seq)

            for frame in tqdm(range(int(seq_dets[:, 0].max()))):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]

                # remove dets with low confidence
                for d in reversed(range(len(dets))):
                    if dets[d, 4] < 0.3:
                        dets = np.delete(dets, d, 0)

                dets[:, 2:4] += dets[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]

                # add label column
                b = np.ones((dets.shape[0], 1))
                dets = np.hstack((dets, b))
                start_time = time.time()
                track_bbs_ids, unm_trk_ext = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time
                total_frames += 1
                # save trackers -> Frame number, ID, Left, Top, Width, Height
                for d in track_bbs_ids:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))