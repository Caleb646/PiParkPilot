import cv2 as cv
import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple, Any, Union, Dict, Optional



import pathlib
import os
import sys

parent = pathlib.Path(os.path.abspath(os.path.curdir))
path = os.path.join(str(parent))
sys.path.append(path)




import pi_park.utils as utils
import pi_park.ssc as ssc

cv.Mat = np.ndarray

@dataclass(frozen=True)
class KPResult:
    all_A_kpts: Optional[np.ndarray] = None
    all_A_desc: Optional[np.ndarray] = None
    all_B_kpts: Optional[np.ndarray] = None
    all_B_desc: Optional[np.ndarray] = None
    matches: Optional[Any] = None
    ssc_A_selected_pts_idxs: Optional[np.ndarray] = None
    ssc_B_selected_pts_idxs: Optional[np.ndarray] = None
    matched_A_kpts: Optional[np.ndarray] = None
    matched_A_desc: Optional[np.ndarray] = None
    matched_B_kpts: Optional[np.ndarray] = None
    matched_B_desc: Optional[np.ndarray] = None


def knn_cross_check_filter(matches, max_pixel_distance=30):
    """
    Parameters
    -----------------------
    `max_pixel_distance`: Respresents the max pixel distance that match A1 can be from its partner A2.
    This is NOT the disparity distance, but rather a measure of how good the match is.
    For example, if A1 is a keypoint located at the top left of a license plate and A2 is located on
    the top right of the same license plate then their `match.distance` would be the pixel width of the license 
    plate. 
    """
    res = []
    for match in matches:
        for m in match:
            if m.distance <= max_pixel_distance:
                res.append(m)
    return res

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
    def __init__(
            self, 
            detector="orb", 
            matcher="knn", 
            cross_check=True,
            scc_num_ret_pts = 30,
            ssc_tolerance = 0.1
            ) -> None:
        self.logger = logging.getLogger(__name__)
        self.cross_check = cross_check
        self.ssc_num_ret_pts = scc_num_ret_pts
        self.ssc_tolerance = ssc_tolerance
        if detector == "sift":
            self.detector = cv.SIFT_create()
            self.matcher = cv.BFMatcher()    
        else:
            self.detector = cv.ORB_create()
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=cross_check) 
            """
            Second param is boolean variable, crossCheck which is false by default. 
            If it is true, Matcher returns only those matches with value (i,j) such 
            that i-th descriptor in set A has j-th descriptor in set B as the best match 
            and vice-versa. That is, the two features in both sets should match each other. 
            It provides consistent result, and is a good alternative to ratio test 
            proposed by D.Lowe in SIFT paper.
            """          
        if matcher == "knn":
            k = 2 if not cross_check else 1
            self.match_f = lambda des1, des2: self.matcher.knnMatch(des1, des2, k=k)
            self.filter_matches = knn_match_filter if not cross_check else knn_cross_check_filter
        else: # Brute force matcher
            self.match_f = lambda des1, des2: self.matcher.match(des1, des2)
            self.filter_matches = bf_match_filter
        self.last = None

    def find_keypoints_def(
            self, left: cv.Mat, right: cv.Mat
            ):
        mask = None
        # Left Image is the query image
        # Right Image is the train image
        left_kpts, left_desc = self.detector.detectAndCompute(left, mask)
        right_kpts, right_desc = self.detector.detectAndCompute(right, mask)
        matches = self.filter_matches(self.match_f(left_desc, right_desc))

        left_matched_kpts = np.asarray([left_kpts[m.queryIdx] for m in matches])
        right_matched_kpts = np.asarray([right_kpts[m.trainIdx] for m in matches])

        left_matched_desc = np.asarray([left_desc[m.queryIdx] for m in matches])
        right_matched_desc = np.asarray([right_desc[m.trainIdx] for m in matches])


        return KPResult(
            all_A_kpts=left_kpts,
            all_A_desc=left_desc,
            all_B_kpts=right_kpts,
            all_B_desc=right_desc,
            matches=matches,
            matched_A_kpts=left_matched_kpts,
            matched_A_desc=left_matched_desc,
            matched_B_kpts=right_matched_kpts,
            matched_B_desc=right_matched_desc
        )
    
    
    def find_keypoints(
            self, left: cv.Mat, right: cv.Mat, use_ssc=False
            ):
        with utils.Timer(self.logger.debug, "KP Matching Time: {} seconds"):
            mask = None
            # Left Image is the query image
            # Right Image is the train image
            left_kpts, left_desc = self.detector.detectAndCompute(left, mask)
            return self.find_keypoints_with_known(
                np.asarray(left_kpts), np.asarray(left_desc), right, use_ssc=use_ssc
                )
        
    def find_keypoints_with_known(
            self, A_kpts, A_desc, B_img: cv.Mat, use_ssc=False
            ):
        mask = None
        # A Image is the query image
        # B Image is the train image
        B_kpts, B_desc = self.detector.detectAndCompute(B_img, mask)
        matches = self.filter_matches(self.match_f(A_desc, B_desc))
        A_matched_kpts = np.asarray([A_kpts[m.queryIdx] for m in matches])
        B_matched_kpts = np.asarray([B_kpts[m.trainIdx] for m in matches])
        """
        OpenCv KeyPoint attributes:
            .pt = (u, v) coordinates of the keypoint
            .response = (float) the response by which the most strong key points 
                have been selected. Can be used for sorting.
            .size = (float) the diameter of the meaningful keypoint neighborhood.
        """
        A_matched_desc = np.asarray([A_desc[m.queryIdx] for m in matches])
        B_matched_desc = np.asarray([B_desc[m.trainIdx] for m in matches])

        A_selected_indices = None
        B_selected_indices = None

        if use_ssc is True:
            height, width = B_img.shape[:2]
            # queryIdx is the index into the original A_kpts array
            A_match_idxs = np.asarray([m.queryIdx for m in matches])
            B_match_idxs = np.asarray([m.trainIdx for m in matches])
            # Sort every kp in A_matched_kpts by response in terms of their index
            sorted_response_idxs = np.argsort(np.asarray([kp.response for kp in A_matched_kpts]), axis=0)[::-1]
            # Use the sorted response indices to sort the query indices
            # This results in an array with a len of len(A_matched_kpts) containing
            # response sorted indices into the original A_kpts array
            A_match_idxs = A_match_idxs[sorted_response_idxs]
            B_match_idxs = B_match_idxs[sorted_response_idxs]

            selected = ssc.ssc(
                # These are the original A key points
                # except they will be sorted by response and only key points
                # matched with B will be sent to ssc
                A_kpts[A_match_idxs],
                self.ssc_num_ret_pts, 
                self.ssc_tolerance,
                width,
                height
                )
            B_kpts = np.asarray(B_kpts)
            B_desc = np.asarray(B_desc)
            A_selected_indices = np.asarray([A_match_idxs[idx] for idx in selected])
            B_selected_indices = np.asarray([B_match_idxs[idx] for idx in selected])

            A_matched_kpts = A_kpts[A_selected_indices]
            B_matched_kpts = B_kpts[B_selected_indices]

            A_matched_desc = A_desc[A_selected_indices]
            B_matched_desc = B_desc[B_selected_indices]
            #return A_selected_indices, A_matched_kpts, A_matched_desc, B_matched_kpts, B_matched_desc
        # NOTE: if use_ssc = True matches will no longer match A_matched_kpts and B_matched_kpts
        return KPResult(
            all_A_kpts=A_kpts,
            all_A_desc=A_desc,
            all_B_kpts=B_kpts,
            all_B_desc=B_desc,
            matches=matches,
            matched_A_kpts=A_matched_kpts,
            matched_A_desc=A_matched_desc,
            matched_B_kpts=B_matched_kpts,
            matched_B_desc=B_matched_desc,
            ssc_A_selected_pts_idxs=A_selected_indices,
            ssc_B_selected_pts_idxs=B_selected_indices
        )
    
        
if __name__ == "__main__":
    matcher = KPMatcher()
    left_1 = cv.imread(f"{path}/data/test_data/KITTI/00/image_0/000000.png")
    left_2 = cv.imread(f"{path}/data/test_data/KITTI/00/image_0/000001.png")
    right_1 = cv.imread(f"{path}/data/test_data/KITTI/00/image_1/000000.png")

    lr_kpres = matcher.find_keypoints_def(left_1, right_1)
    pc_kpres = matcher.find_keypoints_with_known(
        lr_kpres.matched_A_kpts, lr_kpres.matched_A_desc, left_2
        )
    #img = cv.drawMatchesKnn(left_1, l, right_1, r, m, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # img = cv.drawKeypoints(left_1, A_matched_kpts, outImage=None, color=(255, 0, 0))
    # cv.imshow("Image", img)
    # cv.waitKey(5000)
    #print(A_matched_kpts.shape, B_matched_kpts.shape, len(matches))
    matches = sorted(pc_kpres.matches, key = lambda x:x.distance)
    img = cv.drawMatches(
        left_1, 
        pc_kpres.all_A_kpts, 
        left_2, 
        pc_kpres.all_B_kpts, 
        matches[:5], 
        None, 
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    cv.imshow("Selected keypoints", img)
    cv.waitKey(0)


    # matches, A_selected_indices, A_matched_kpts, A_matched_desc, B_matched_kpts, B_matched_desc = matcher.find_keypoints(left_1, right_1, use_ssc=True)
    # img = cv.drawKeypoints(left_1, A_matched_kpts, outImage=None, color=(255, 0, 0))
    # cv.imshow("Selected keypoints", img)
    # cv.waitKey(0)

    cv.destroyAllWindows()