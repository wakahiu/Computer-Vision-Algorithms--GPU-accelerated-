#include "commonHeaders.h"

int main(int, char**)
{
    VideoCapture cap(0); 			// open the default camera
    assert( cap.isOpened());  		// check if we succeeded

	cl_int err;
	cl::vector< Platform > platformList;
	cl::Platform::get(&platformList);
	
	namedWindow("GPU",CV_WINDOW_AUTOSIZE);
	while(true){
		Mat frame;
        cap >> frame; // get a new frame from camera
        
        int key = waitKey(30);	
     	if( key == ESC) break;
     	
     	imshow("GPU", frame);

	}
    return 0;
}
