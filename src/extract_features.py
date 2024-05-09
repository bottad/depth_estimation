import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import argparse
import json

from include.path_manager import *

class Image_handler:

    def __init__(self, featuretype, matchingtype, number = 1000, outname = "new", showImages = False): 
        self.featuretype = featuretype
        self.matchingtype = matchingtype
        # Load Images 
        self.in_path = read_path_from_file("input-path")
        left_imgPath = os.path.join(self.in_path, 'Stereo_images_left', 'left_image_{}.jpg'.format(number))
        right_imgPath = os.path.join(self.in_path, 'Stereo_images_right', 'right_image_{}.jpg'.format(number))
        self.left_imgGray = cv.imread(left_imgPath, cv.IMREAD_GRAYSCALE)
        self.right_imgGray = cv.imread(right_imgPath, cv.IMREAD_GRAYSCALE)
        self.outname = outname

        with open("data/camera_parameters.json", "r") as file:
            camera_parameters = json.load(file)
            self.BASELINE = camera_parameters["BASELINE"]
            self.FOCAL_LENGTH = camera_parameters["FOCAL_LENGTH"]

        if showImages: 
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.left_imgGray)
            plt.subplot(122)
            plt.imshow(self.right_imgGray)
            plt.show()

    def extract_features(self):
        print("[INFO]\tExtracting features...")
        if self.featuretype == "SIFT":
            self.get_SIFT_features()
        elif self.featuretype == "AKAZE":
            self.get_AKAZE_features()
        else:
            print(Fore.RED + "[ERROR]\tFeature type not available")
            return
        print("[INFO]\t... done!")

    def match_features(self):
        print("[INFO]\tMatching features...")
        if self.matchingtype == "Bruteforce":
            self.match_features_bruteforce()
        elif self.matchingtype == "FLANN":
            self.match_features_FLANN()
        elif self.matchingtype == "FLANN-epipolar":
            self.match_features_FLANNepipolar()
        else:
            print(Fore.RED + "[ERROR]\tMatching type not available")
            return
        print("[INFO]\t... done!")

    def get_SIFT_features(self, show = False): 

        sift = cv.SIFT_create()
        self.left_keypoints, self.left_descriptors = sift.detectAndCompute(self.left_imgGray,None)
        self.right_keypoints, self.right_descriptors = sift.detectAndCompute(self.right_imgGray, None)

        if show:
            left_imgFeatures = cv.drawKeypoints(self.left_imgGray,self.left_keypoints,self.left_imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            right_imgFeatures = cv.drawKeypoints(self.right_imgGray,self.right_keypoints,self.right_imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.figure() 
            plt.subplot(121)
            plt.imshow(left_imgFeatures)
            plt.subplot(122)
            plt.imshow(right_imgFeatures)
            plt.show()

    def get_AKAZE_features(self, show = False): 

        sift = cv.AKAZE_create()
        self.left_keypoints, self.left_descriptors = sift.detectAndCompute(self.left_imgGray,None)
        self.right_keypoints, self.right_descriptors = sift.detectAndCompute(self.right_imgGray, None)

        if show:
            left_imgFeatures = cv.drawKeypoints(self.left_imgGray,self.left_keypoints,self.left_imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            right_imgFeatures = cv.drawKeypoints(self.right_imgGray,self.right_keypoints,self.right_imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.figure() 
            plt.subplot(121)
            plt.imshow(left_imgFeatures)
            plt.subplot(122)
            plt.imshow(right_imgFeatures)
            plt.show()

    def match_features_bruteforce(self, show=True):
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(self.left_descriptors, self.right_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Adjust the threshold as needed
                good_matches.append(m)

        if show:
            imgMatch = cv.drawMatches(self.left_imgGray, self.left_keypoints, self.right_imgGray, self.right_keypoints, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Save the image with matched features
            cv.imwrite(os.path.join('out', self.outname + '_matched_' + self.featuretype + '_bruteforce.jpg'), imgMatch)

            plt.imshow(imgMatch)
            plt.title('Matched Features')
            plt.show()

        # Compute distances to matched points
        with open(os.path.join('out', self.outname + '_features.csv'), 'w') as f:
            for match in good_matches:
                left_pt = self.left_keypoints[match.queryIdx].pt
                right_pt = self.right_keypoints[match.trainIdx].pt
                distance_px = abs(left_pt[0] - right_pt[0])  # pixel disparity

                # Compute depth and write to file
                depth_in_meters = (self.BASELINE * self.FOCAL_LENGTH) / distance_px  # Convert pixel disparity to mm
                f.write(f"{left_pt[0]},{left_pt[1]},{depth_in_meters}\n")

    def match_features_FLANN(self, show=True): 
        FLANN_INDEX_KDTREE = 1
        nKDtrees = 5 
        nLeafChecks = 50
        nNeighbors = 2 
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDtrees)
        searchParams = dict(checks=nLeafChecks) 
        flann = cv.FlannBasedMatcher(indexParams, searchParams)
        matches = flann.knnMatch(self.left_descriptors, self.right_descriptors, k=nNeighbors)
        matchesMask = [[0,0] for _ in range(len(matches))]
        testRatio = 0.7 # ratio test
        for i, (m, n) in enumerate(matches):
            if m.distance < testRatio * n.distance:
                matchesMask[i] = [1,0]

        if show:
            drawParams = dict(matchColor=(0,255,0), singlePointColor=(255,0,0),
                matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
            imgMatch = cv.drawMatchesKnn(self.left_imgGray, self.left_keypoints, self.right_imgGray, self.right_keypoints, matches, None, **drawParams)

            # Save the image with matched features
            cv.imwrite(os.path.join('out', self.outname + '_matched_' + self.featuretype + '_FLANN.jpg'), imgMatch)

            plt.figure() 
            plt.imshow(imgMatch)
            plt.title('Matched Features')
            plt.show()

        # Compute distances to matched points
        self.disparity = []
        for match_pair in matches:
            # Unpack the queryIdx and trainIdx from the first match in the pair
            queryIdx = match_pair[0].queryIdx
            trainIdx = match_pair[0].trainIdx
            left_pt = self.left_keypoints[queryIdx].pt
            right_pt = self.right_keypoints[trainIdx].pt
            distance_px = abs(left_pt[0] - right_pt[0])  # pixel disparity
            self.disparity.append(distance_px)

        with open(os.path.join('out', self.outname + '_features.csv'), 'w') as f:
            for i, (m, n) in enumerate(matches):
                if matchesMask[i][0] == 1:  # Check if the match passed the filtering criteria
                    # Unpack the queryIdx and trainIdx from the first match in the pair
                    queryIdx = m.queryIdx
                    trainIdx = m.trainIdx
                    left_pt = self.left_keypoints[queryIdx].pt
                    right_pt = self.right_keypoints[trainIdx].pt
                    distance_px = abs(left_pt[0] - right_pt[0])  # pixel disparity

                    # Compute depth and write to file
                    depth_in_meters = (self.BASELINE * self.FOCAL_LENGTH) / distance_px  # Convert pixel disparity to mm
                    f.write(f"{left_pt[0]},{left_pt[1]},{depth_in_meters}\n")
    
    def match_features_FLANNepipolar(self, num_features=200, show=True):
        FLANN_INDEX_KDTREE = 1
        nKDtrees = 5
        nLeafChecks = 50
        nNeighbors = 2
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDtrees)
        searchParams = dict(checks=nLeafChecks)
        flann = cv.FlannBasedMatcher(indexParams, searchParams)
        matches = flann.knnMatch(self.left_descriptors, self.right_descriptors, k=nNeighbors)
        matchesMask = [[0,0] for _ in range(len(matches))]
        testRatio = 0.65  # ratio test

        for i, (m, n) in enumerate(matches):
            if m.distance < testRatio * n.distance:
                # Get the keypoints for the matches
                left_pt = self.left_keypoints[m.queryIdx].pt
                right_pt = self.right_keypoints[m.trainIdx].pt
                # Check if the matched keypoints are roughly on a horizontal line
                if abs(left_pt[1] - right_pt[1]) < 20:  # Adjust the threshold as needed
                    matchesMask[i] = [1, 0]

        if show:
            drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                              matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
            imgMatch = cv.drawMatchesKnn(self.left_imgGray, self.left_keypoints, self.right_imgGray, self.right_keypoints, matches, None, **drawParams)

            # Save the image with matched features
            cv.imwrite(os.path.join('out', self.outname + '_matched_' + self.featuretype + '_FLANN-epipolar.jpg'), imgMatch)

            plt.figure()
            plt.imshow(imgMatch)
            plt.title('Matched Features')
            plt.show()

        with open(os.path.join('out', self.outname + '_features.csv'), 'w') as f:
            for i, (m, n) in enumerate(matches):
                if matchesMask[i][0] == 1:  # Check if the match passed the filtering criteria
                    # Unpack the queryIdx and trainIdx from the first match in the pair
                    queryIdx = m.queryIdx
                    trainIdx = m.trainIdx
                    left_pt = self.left_keypoints[queryIdx].pt
                    right_pt = self.right_keypoints[trainIdx].pt
                    distance_px = abs(left_pt[0] - right_pt[0])  # pixel disparity

                    # Compute depth and write to file
                    depth_in_meters = (self.BASELINE * self.FOCAL_LENGTH) / distance_px  # Convert pixel disparity to mm
                    f.write(f"{left_pt[0]},{left_pt[1]},{depth_in_meters}\n")

    def generate_depth_prior_image(self):
        image = cv.cvtColor(self.left_imgGray, cv.COLOR_GRAY2BGR)  # Convert grayscale to BGR for drawing color circles
        with open(os.path.join('out', self.outname + '_features.csv'), 'r') as file:
            for line in file:
                x, y, depth = map(float, line.strip().split(','))
                cv.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)  # Draw small red circle for feature point
                cv.putText(image, f'{depth:.2f}', (int(x) + 5, int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Display depth beside the feature point
        
        cv.imwrite(os.path.join('out', self.outname + '_priors_' + self.featuretype + '_' + self.matchingtype + '.jpg'), image)
        
        # Display the image with feature points and depth
        cv.imshow('Image with Feature Points and Depth', image)
        cv.waitKey(0)
        cv.destroyAllWindows()

#######################################################################################
###                              main                                               ###
#######################################################################################

def main():

    featuretype = "SIFT"
    matchtype = "Bruteforce"

    # Parse input arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inpath', action='store', type=str, help='Set a new input path to the folder containing the left and right image-folders')
    parser.add_argument('-n', '--number', action='store', type=str, help='Define the image number that should be used')
    parser.add_argument('-o', '--outname', action='store', type=str, help='Define the output name')
    parser.add_argument('-f', '--featuretype', action='store', type=str, help='Choose feauture type: s -> SIFT, a -> AKAZE')
    parser.add_argument('-m', '--matchtype', action='store', type=str, help='Choose feauture matching type: b -> Bruteforce, f -> FLANN, fe -> FLANN only allowing epipolar matches')

    args = parser.parse_args()

    if args.inpath is not None:
        print(Fore.GREEN + "Reseting image input path to: " + Style.RESET_ALL, args.inpath)
        # Save the input path to the image folders
        save_path_to_file(args.inpath, "input-path")

    if args.number is not None:
        print(Fore.GREEN + "Using images with number: " + Style.RESET_ALL, args.number)
        
        number = args.number

    if args.outname is not None:
        print(Fore.GREEN + "Name of the output file: " + Style.RESET_ALL, args.outname)
        
        outname = args.outname

    if args.featuretype is not None:
        if args.featuretype == "s":
            print(Fore.GREEN + "Using SIFT features" + Style.RESET_ALL)
        elif args.featuretype == "a":
            print(Fore.GREEN + "Using AKAZE features" + Style.RESET_ALL)
            featuretype = "AKAZE"
        else:
            print(Fore.YELLOW + "No valid feature type chosen -> using default feature type (SIFT)" + Style.RESET_ALL)
    else: 
        # Default
        print(Fore.GREEN + "Using default feature type (SIFT)" + Style.RESET_ALL)

    if args.matchtype is not None:
        if args.matchtype == "b":
            print(Fore.GREEN + "Using Bruteforce matching method" + Style.RESET_ALL)
        elif args.matchtype == "f":
            print(Fore.GREEN + "Using FLANN matching method" + Style.RESET_ALL)
            if featuretype == "AKAZE":
                print("[WARNING]\t" + Fore.YELLOW + "FLANN does not work with AKAZE, using Bruteforce instead!" + Style.RESET_ALL)
            else:
                matchtype = "FLANN"
        elif args.matchtype == "fe":
            print(Fore.GREEN + "Using FLANN matching method only allowing epipolar matches" + Style.RESET_ALL)
            if featuretype == "AKAZE":
                print("[WARNING]\t" + Fore.YELLOW + "FLANN-epipolar does not work with AKAZE, using Bruteforce instead!" + Style.RESET_ALL)
            else:
                matchtype = "FLANN-epipolar"
        else:
            print(Fore.YELLOW + "No valid matching type chosen -> using default matching type (Bruteforce)" + Style.RESET_ALL)
    else:
        # Default
        print(Fore.GREEN + "Using default matching type (Bruteforce)" + Style.RESET_ALL)

    image_handler = Image_handler(featuretype=featuretype, matchingtype=matchtype, number=number, outname=outname)

    # Run depth estimation
    image_handler.extract_features()
    image_handler.match_features()
    image_handler.generate_depth_prior_image()



if __name__ == '__main__': 
    main()