import cv2 as cv
import numpy as np
import logging
import time

from typing import List, Tuple, Any, Union, Dict

cv.Mat = np.ndarray

def knn_match_filter(matches, d=0.7):
    """
    Parameters
    ------------
    matches -> List[Tuple(DMatch, DMatch)] there are k DMatches inside the tuple.
        - DMatch.distance - Distance between descriptors. The lower, the better it is.
        - DMatch.trainIdx - Index of the descriptor in train descriptors
        - DMatch.queryIdx - Index of the descriptor in query descriptors
        - DMatch.imgIdx - Index of the train image.
    """
    ret = []
    for m, n in matches:
        if m.distance < d * n.distance:
            #ret.append((m, n))
            ret.append(m)
    return ret

def bf_match_filter(matches: List[Any], top=2_000):
    """
    Parameters
    ------------
    matches -> List[DMatch]
        - DMatch.distance - Distance between descriptors. The lower, the better it is.
        - DMatch.trainIdx - Index of the descriptor in train descriptors
        - DMatch.queryIdx - Index of the descriptor in query descriptors
        - DMatch.imgIdx - Index of the train image.
    """
    matches.sort(key=lambda x:x.distance)
    return matches[:min(len(matches), top)]

class KPMatcher:
    def __init__(self, detector="orb", matcher="knn") -> None:
        self.logger = logging.getLogger(__name__)
        if detector == "sift":
            self.detector = cv.SIFT_create()
            self.matcher = cv.BFMatcher()    
        else:
            self.detector = cv.ORB_create()
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING)#, crossCheck=True)           
        if matcher == "knn":
            self.match_f = lambda des1, des2: self.matcher.knnMatch(des1, des2, k=2)
            self.filter_matches = knn_match_filter
        else: # Brute force matcher
            self.match_f = lambda des1, des2: self.matcher.match(des1, des2)
            self.filter_matches = bf_match_filter
        self.last = None

    def find_keypoints(
            self, left: cv.Mat, right: cv.Mat
            ):
        mask = None
        left_kpts, left_desc = self.detector.detectAndCompute(left, mask)
        right_kpts, right_desc = self.detector.detectAndCompute(right, mask)
        matches = self.filter_matches(self.match_f(left_desc, right_desc))
        left_matched_pts = np.asarray([left_kpts[m.queryIdx].pt for m in matches])
        right_matched_pts = np.asarray([right_kpts[m.trainIdx].pt for m in matches])
        return matches, left_matched_pts, right_matched_pts
    
        
if __name__ == "__main__":
    matcher = KPMatcher()
    left_1 = cv.imread("./data/test_data/kp_matching/f0001.jpg")
    right_1 = cv.imread("./data/test_data/kp_matching/f0002.jpg")
    left_2 = cv.imread("./data/test_data/kp_matching/f0003.jpg")
    right_2 = cv.imread("./data/test_data/kp_matching/f0004.jpg")

    matcher.find_keypoints(left_1, right_1)
    matcher.find_keypoints(left_2, right_2)