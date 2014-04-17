#include "commonHeaders.h"


int main()
{

	String img1("panorama_image1.jpg");
	String img2("panorama_image2.jpg");
	
	Mat image1 = imread(img1.c_str());
	Mat image2 = imread(img2.c_str());
	
	if(!image1.data || !image2.data){
		cerr << "Could not open or find the image" << endl;
	}
	
	Mat gray_image1;
	Mat gray_image2;
	
	//Convert the images to gray scale
	cvtColor(image1,gray_image1,CV_BGR2GRAY,0);
	cvtColor(image2,gray_image2,CV_BGR2GRAY,0);
	
	/*
	unsigned char * imgPtr1 = (unsigned char *)image1.data;
	
	for(int j = 0; j < image1.rows; j++){
		for(int i = 0; i <image1.cols*3; i+=3){
			int b = imgPtr1[image1.step*j+i];
			int g = imgPtr1[image1.step*j+i+1];
			int r = imgPtr1[image1.step*j+i+2];
			int gscale = (b+g+r)/3;
			
			imgPtr1[image1.step*j+i] = gscale;
			imgPtr1[image1.step*j+i+1] = gscale;
			imgPtr1[image1.step*j+i+2] = gscale;
		}
	}
	*/
	namedWindow(img1.c_str(),WINDOW_AUTOSIZE);		//Create a window for display
	namedWindow(img2.c_str(),WINDOW_AUTOSIZE);		//Create a window for display
	
	imshow(img1.c_str(),gray_image1);
	imshow(img2.c_str(),gray_image2);

	//
	// -- Step 1: Detect the keypoints using SURF detector
	int minHessian = 50;
	
	//SurfFeatureDetector detector(minHessian);
	
	vector<KeyPoint> keyPoints;
	
	//detector.detect(gray_image1,keyPoints);
	
	
		
	waitKey(0);
    return 0;
}
