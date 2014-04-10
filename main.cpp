#include "commonHeaders.h"


int main()
{
	cl_int err;
    VideoCapture cap(0); 					// open the default camera
    assert( cap.isOpened());  				// check if we succeeded

	/*
	* Create a context
	*/
	
	const int attributeCount = 5;
	const cl_platform_info attributeTypes[attributeCount] = 
												{	CL_PLATFORM_NAME, 
													CL_PLATFORM_VENDOR,
													CL_PLATFORM_VERSION, 
													CL_PLATFORM_PROFILE, 
													CL_PLATFORM_EXTENSIONS };
	string attributeNames[attributeCount] = 
								{ 	string("Name \t"), 
									string("Vendor \t"),
        							string("Version \t"), 
        							string("Profile \t"),
        							string("Extensions \t")};
	
	vector< Platform > platformList;		//List of platforms
	Platform::get(&platformList);			//Get the list of platforms
	
	//Check if we have a platform
	if(platformList.size()==0){
		cerr << "No platforms found. Check OpenCL installation" << endl;
		exit(EXIT_FAILURE);
	}
	
	for(int i = 0; i < platformList.size() ; i++){

		cout << "\n-----------------------------------------------------------"<<endl;
		cout << "\t---Platform Attributes---" << endl;
        for (int j = 0; j < attributeCount; j++) {

            // get platform attribute value size
            string platformInfo;
			platformList[i].getInfo(attributeTypes[j],&platformInfo);
			cout << attributeNames[j] << platformInfo << endl;
        }
        
        
		//See the devices available on our platform.
		vector<Device> devices;
		platformList[i].getDevices(CL_DEVICE_TYPE_ALL,&devices);
		if(devices.size()==0){
			cerr << "No devices found. Check OpenCL installation" << endl;
			exit(EXIT_FAILURE); 
		}
		cout << "\n\t---Devices on this platform---" << endl;
		for (int j = 0; j < devices.size(); j++) {
            // get device attribute value size
            cout << "\tDevice name " << devices[j].getInfo<CL_DEVICE_NAME>() << endl;
        }
	}
	
	/*
	context_properties clProps[3] = {	CL_CONTEXT_PLATFORM,
											(cl_context_properties)(platformList[0])(),
											0	
											};
	Context context(CL_DEVICE_TYPE_CPU,
					clProps,
					NULL,
					NULL,
					&err);
	CL_CHECKERROR(err); 
	*/										
									
	 
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platforms;
	//cl::Platform::get(&platformList);
	
	CL_CHECKERROR( clGetPlatformIDs(1, &platform_id, &ret_num_platforms));
	
	
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
