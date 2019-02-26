#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;


int main(){
	Mat img;
	img = imread("/home/xuesong/RailSurfaceInspection/3142.jpg");
	if(img.empty()){
		cout << "image is not read"	<< endl;
	}
	//namedWindow("test");
	//imshow("test", img);
	//waitKey();
	
	return 0;
}
