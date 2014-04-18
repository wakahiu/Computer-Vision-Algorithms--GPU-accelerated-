#include "commonHeaders.h"

//Converts the image to gray scale using equal weighting.
Mat toGray(Mat img){
	
	Mat grayImg = img.clone();
	unsigned char * imgPtr = (unsigned char *)img.data;
	unsigned char * grayPtr = (unsigned char *)grayImg.data;
	
	for(int j = 0; j < img.rows; j++){
		for(int i = 0; i <img.cols*3; i+=3){
		
			int b = imgPtr[img.step*j+i];
			int g = imgPtr[img.step*j+i+1];
			int r = imgPtr[img.step*j+i+2];
			
			int gscale = (b+g+r)/3;
			
			grayPtr[img.step*j+i] = gscale;
			grayPtr[img.step*j+i+1] = gscale;
			grayPtr[img.step*j+i+2] = gscale;
		}
	}
	return grayImg;
	
}

int main()
{

	String img1("panorama_image1.jpg");
	String img2("panorama_image2.jpg");
	
	Mat image1 = imread(img1.c_str());
	Mat image2 = imread(img2.c_str());
	
	if(!image1.data || !image2.data){
		cerr << "Could not open or find the image" << endl;
	}
	
	//Convert the images to gray scale
	Mat gray_image1 = toGray(image1);
	Mat gray_image2 = toGray(image2);
	
	//namedWindow(img1.c_str(),WINDOW_AUTOSIZE);		//Create a window for display
	//namedWindow(img2.c_str(),WINDOW_AUTOSIZE);		//Create a window for display

	//
	// -- Step 1: Detect the keypoints using SURF detector
	int minHessian = 400;
	
	SurfFeatureDetector detector(minHessian);
	
	vector<KeyPoint> keyPoints_object;
	vector<KeyPoint> keyPoints_scene;
	
	detector.detect(gray_image1,keyPoints_object);
	detector.detect(gray_image2,keyPoints_scene);
	
	
	//Calculate descriptors (feature vectors);
	SurfDescriptorExtractor extractor;
	
	Mat descriptors_object;
	Mat descriptors_scene;
	
	extractor.compute(gray_image1,keyPoints_object, descriptors_object);
	extractor.compute(gray_image2,keyPoints_scene, descriptors_scene);
	
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_object, descriptors_scene,matches);
	
	double maxDist = 0.0;
	double minDist = 100.0;
	
	//Quick caluculation of max and min distances between keypoints
	for(int i = 0; i < descriptors_object.rows; i++){
		double dist  = matches[i].distance;
		minDist = dist < minDist ? dist : minDist;
		maxDist = dist > maxDist ? dist : maxDist;	
	}
	
	cout << "Maximum Distance: " << maxDist << endl;
	cout << "Minimum Distance: " << minDist << endl;
	
	//Use only good matches
	double distFilter = 3.0;
	vector<DMatch> goodMatches;
	for(int i = 0; i < descriptors_object.rows; i++){
		if( matches[i].distance  < minDist * distFilter ){
			goodMatches.push_back(matches[i]);
		}
	}
	
	vector<Point2f> obj;
	vector<Point2f> scn;
	
	for(int i = 0; i < goodMatches.size(); i++){
		obj.push_back(keyPoints_object[goodMatches[i].queryIdx].pt );
		scn.push_back(keyPoints_scene[goodMatches[i].trainIdx].pt );
	}
	
	//Find the homography matrix
	Mat H = findHomography(obj,scn,CV_RANSAC);
	
	Mat result;
	warpPerspective(image1,result,H,Size(image1.cols+image2.cols,image1.rows));
	Mat half(result,Rect(0,0,image2.cols,image2.rows));
	
	image2.copyTo(half);
	imshow("Result",half);
	
	waitKey(0);
    return 0;
}
