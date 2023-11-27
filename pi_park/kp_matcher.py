import cv2 as cv
import numpy as np

from typing import List, Tuple, Any, Union, Dict

class KPMatcher:
    def __init__(self) -> None:
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        self.last = None

    def get_keypoints_descriptors_(
            self, img: Union[cv.Mat, Dict[str, List[Any]]]
            ) -> Tuple[List[Any], List[Any]]:
        if isinstance(img, dict):
            return img["kpts"], img["desc"]
        else:
            image = np.mean(img, axis=2).astype(np.uint8) if len(img.shape) > 2 else img.astype(np.uint8)
            pts = cv.goodFeaturesToTrack(
                image=image, maxCorners=4500, qualityLevel=0.02, minDistance=3
                )
            kpts = [cv.KeyPoint(p[0][0], p[0][1], size=30) for p in pts]
            kpts, descriptors = self.orb.compute(img, kpts)
            return kpts, descriptors

    def find_keypoints(
            self, left: Union[cv.Mat, Dict[str, List[Any]]], right: Union[cv.Mat, Dict[str, List[Any]]]
            ):
        all_left_kpts, all_left_descriptors = self.get_keypoints_descriptors_(left)
        all_right_kpts, all_right_descriptors = self.get_keypoints_descriptors_(right)
        ret = []
        matches = self.bf.knnMatch(all_left_descriptors, all_right_descriptors, k=2)
        for m, n in matches:
            if m.distance < 0.5 * n.distance: # and m.distance < 64:
                ret.append((m, n))
        matches = ret
        left_matched_pts = np.asarray([all_left_kpts[m.queryIdx].pt for m, n in matches])
        right_matched_pts = np.asarray([all_right_kpts[m.trainIdx].pt for m, n in matches])  
        #homography, mask = cv.findHomography(left_matched_pts, right_matched_pts, cv.RANSAC, 100.0)
        #mask = mask.ravel()

        # matched points used in homography
        # left_used_pts = left_matched_pts[mask == 1]
        # right_used_pts = right_matched_pts[mask == 1]

        #self.last = {"kpts": kpts, "descriptors": descriptors}
        #return pts1.T, pts2.T, kpts, ret
        #return all_left_kpts, all_left_descriptors, left_used_pts.T, right_used_pts.T
        return matches, left_matched_pts, right_matched_pts
    
    # def find_keypoints(self, left: cv.Mat, right: cv.Mat):
    #     # find the keypoints that the left and right image share
    #     all_left_keypts, all_left_desc, left_pts, right_pts = self.find_keypoints_(left, right)
    #     print(f"Matched left: {left_pts.shape}")
    #     # find the keypoints from the current and previous frames that match
    #     if self.last is not None:
    #         all_left_keypts, all_left_desc, left_pts, right_pts = self.find_keypoints_(
    #             self.last,
    #             dict(kpts=all_left_keypts, desc=all_left_desc)
    #             )
    #         self.last = dict(kpts=all_left_keypts, desc=all_left_desc)
    #         print(f"Matched points from previous: {left_pts.shape}")
    #         return left_pts, right_pts
    #     else:
    #         self.last = dict(kpts=all_left_keypts, desc=all_left_desc)
    #         return left_pts, right_pts
    
        
if __name__ == "__main__":
    matcher = KPMatcher()
    left_1 = cv.imread("./data/test_data/kp_matching/f0001.jpg")
    right_1 = cv.imread("./data/test_data/kp_matching/f0002.jpg")
    left_2 = cv.imread("./data/test_data/kp_matching/f0003.jpg")
    right_2 = cv.imread("./data/test_data/kp_matching/f0004.jpg")

    matcher.find_keypoints(left_1, right_1)
    matcher.find_keypoints(left_2, right_2)