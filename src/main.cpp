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

//Plot a green dot around point (x,y)
void plot(Mat img,int x,int y){
	
	int k=1;
	unsigned char * imgPtr = (unsigned char *)img.data;
	
	for(int j = y-k; j < y+k; j++){
		for(int i = x-k; i < x+k; i++ ){
		
			//Bounds check
			if( j < 0 || j >= img.rows || i < 0 || i >= img.cols){
				continue;
			}
	
			imgPtr[img.step*j+i*3+0] = 0;
			imgPtr[img.step*j+i*3+1] = 255;
			imgPtr[img.step*j+i*3+2] = 0;
		}
	}
	
}


void findSIFTPoints(Mat gray_img, float * DOGKernel,int thr,int hw, int stride){

	int w = 2*hw+1;
	unsigned char * imgPtr = (unsigned char *)gray_img.data;
	
	
	for(int j = 0+hw; j < (gray_img.rows-hw); j+=stride){
		for(int i = 0+hw; i <(gray_img.cols-hw)*3; i+=(3*stride)){
		
			
			float c =0;
			for(int x = -hw; x <= hw;x++){
				for(int y = -hw; y <= hw ;y++){
					float g = DOGKernel[ (x+hw)*w + y + hw];
					c += imgPtr[gray_img.step*(j+x)+(i+y)]*g;
				}
			}
			if(c > thr){
				plot(gray_img,i/3,j);
			}
			
		}
	}
}

Mat GaussianBlurr(Mat img, float * GaussKernel,int hw){

	Mat blurr_img = img.clone();
	
	int w = 2*hw+1;
	unsigned char * imgPtr = (unsigned char *)img.data;
	unsigned char * blurrPtr = (unsigned char *)blurr_img.data;
	
	for(int j = 0+hw; j < (img.rows-hw); j++){
		for(int i = 0+hw; i <(img.cols-hw)*3; i+=3){
		
			
			float b =0;
			float g =0;
			float r =0;
			
			for(int y = -hw; y <= hw; y++){
				for(int x = -hw; x <= hw ;x++){
					float k = GaussKernel[ (y+hw)*w + x + hw];
					b += imgPtr[img.step*(j+y)+(i+x)+0]*k;
					g += imgPtr[img.step*(j+y)+(i+x)+1]*k;
					r += imgPtr[img.step*(j+y)+(i+x)+2]*k;
				}
			}
			
			blurrPtr[blurr_img.step*j+i+0] = b;
			blurrPtr[blurr_img.step*j+i+1] = g;
			blurrPtr[blurr_img.step*j+i+2] = r;
			
			
		}
	}
	return blurr_img;
}	

	
int main()
{

	String img1("panorama_image4.jpg");
	String img2("panorama_image2.jpg");
	
	Mat image1 = imread(img1.c_str());
	Mat image2 = imread(img2.c_str());
	
	if(!image1.data || !image2.data){
		cerr << "Could not open or find the image" << endl;
	}
	
	//Convert the images to gray scale
	Mat gray_image1 = toGray(image1);
	Mat gray_image2 = toGray(image2);
	
	
	//Difference of gaussians (DOG) kernel.
	int hw = 7;
	int w = 2*hw+1;
	float s = 1.8;
	float sigmaSq = 3.0;
	float DOGKernel[w][w];
	
	for(int i = -hw; i <= hw ;i++){
		for(int j = -hw; j <= hw ;j++){
		
			float g = (1/(2*M_PI*sigmaSq*s*s))*exp( -(i*i + j*j)/(2*sigmaSq*s*s) );
			float h = (1/(2*M_PI*sigmaSq))*exp( -(i*i + j*j)/(2*sigmaSq) );
			
			DOGKernel[i+hw][j+hw] = (g - h)/(s-1);
		}
	}
	
	float gaussKernel[w][w];
	float total = 0.0;
	
	sigmaSq = 15.0;
	//Gaussian Kernel
	for(int i = -hw; i <= hw ;i++){
		for(int j = -hw; j <= hw ;j++){
		
			float g = exp( -(i*i + j*j)/(2*sigmaSq) );
			
			gaussKernel[i+hw][j+hw] = g;
			total += g;
		}
	}
	//Normalize the gaussian Kernel
	for(int i = -hw; i <= hw ;i++){
		for(int j = -hw; j <= hw ;j++){
		
			gaussKernel[i+hw][j+hw] /= total;
			//cout << gaussKernel[i+hw][j+hw] << "\t" ;
		}
		//cout << endl;
	}
	
	Mat blurr_gray_image1 = GaussianBlurr(gray_image1,&gaussKernel[0][0],hw);
	Mat blurr_gray_image2 = GaussianBlurr(gray_image2,&gaussKernel[0][0],hw);
	
	
	int thr = 6;
	int stride = 3;
	findSIFTPoints(blurr_gray_image1, &DOGKernel[0][0], thr, hw, stride);
	findSIFTPoints(blurr_gray_image2, &DOGKernel[0][0], thr, hw, stride);
	
	namedWindow(img1.c_str(),WINDOW_AUTOSIZE);		//Create a window for display
	namedWindow(img2.c_str(),WINDOW_AUTOSIZE);		//Create a window for display
	imshow(img1.c_str(),blurr_gray_image1);
	imshow(img2.c_str(),blurr_gray_image2);
	
	waitKey(0);
	return 0;
	
	
	
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
