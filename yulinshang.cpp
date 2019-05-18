#include "opencv2/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <vector>  

using namespace cv;
using namespace std;

RNG rng(12345);

void log_normalization(Mat& Img)
{
	Img = (Img + 1)*0.5;
	Mat logImg;
	cout << logImg << endl;
	log(Img, logImg);

	Mat tmp_m, tmp_std;
	double meanV, stdV;
	meanStdDev(logImg, tmp_m, tmp_std);
	meanV = tmp_m.at<double>(0, 0);
	stdV = tmp_std.at<double>(0, 0);

	logImg = (logImg - meanV) / stdV;
	normalize(logImg, Img, 1.0, 0.0, NORM_MINMAX);
}


void contourDetector(Mat& Img){
	// Rect bounding_rect;
    vector<vector<Point> > contours; // Vector for storing contours
	vector<Vec4i> hierarchy;

    findContours(Img, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE ); // Find the contours in the image
	cout << "contour number: " << contours.size() << endl;
	
	Mat drawing = Mat::zeros(Img.size(), CV_8UC3);

	int num = 0;

    for(auto it = contours.begin(); it != contours.end(); it++){ // iterate through each contour.
        double area = contourArea(*it);  //  Find the area of contour
		if(area > 5){
			cout << area << endl;
			num ++;
		}
		else{
			contours.erase(it);	
		}
		
    }
	cout << "contours: " << contours[2] << endl;
	

	for( int i = 0; i < contours.size(); i++ ){       
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    	drawContours(drawing, contours, i, Scalar( 0, 255, 0 ), 1 );     
	}

	cout << "yulin number: " << num << endl;
	namedWindow("contour");
	imshow( "contour", drawing);
}


void connected_component_stats_demo(Mat &image) {

    // 二值化
    //Mat gray, binary;
    //cvtColor(image, gray, COLOR_BGR2GRAY);
    //threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat binary;	
	medianBlur(image, image, 3);
	adaptiveThreshold(image, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 3, 10);
	imshow("bianry", binary);

    // 形态学操作
    Mat k = getStructuringElement(MORPH_RECT, Size(8, 4), Point(-1, -1));
    //morphologyEx(binary, binary, MORPH_OPEN, k);
    morphologyEx(binary, binary, MORPH_CLOSE, k);
    imshow("binary", binary);
    Mat labels = Mat::zeros(image.size(), CV_32S);
    Mat stats, centroids;
    int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
    //printf("total labels : %d\n", (num_labels - 1));
    vector<Vec3b> colors(num_labels);
	

    // background color
    colors[0] = Vec3b(0, 0, 0);

    // object color
    int b = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int r = rng.uniform(0, 256);
    for (int i = 1; i < num_labels; i++) {
        colors[i] = Vec3b(0, 255, 0);
    }

    // render result
    Mat dst = Mat::zeros(image.size(), image.type());
    int w = image.cols;
    int h = image.rows;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int label = labels.at<int>(row, col);
            if (label == 0) continue;
            dst.at<Vec3b>(row, col) = colors[label];
        }
    }

    for (int i = 1; i < num_labels; i++) {
        Vec2d pt = centroids.at<Vec2d>(i, 0);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);
        int area = stats.at<int>(i, CC_STAT_AREA);
        printf("area : %d, center point(%.2f, %.2f)\n", area, pt[0], pt[1]);
        circle(dst, Point(pt[0], pt[1]), 2, Scalar(0, 0, 255), -1, 8, 0);
        rectangle(dst, Rect(x, y, width, height), Scalar(255, 0, 255), 1, 8, 0);
    }
    imshow("ccla-demo", dst);
	waitKey();
}


int main(){
	double t0 = (double)getTickCount();

	Mat img = imread("/home/xuesong/RailInspection/defectsImages/yulinshang/y1.png",0);
	
	namedWindow("raw");

	namedWindow("binary");
	namedWindow("connection");
	namedWindow("blur");
	imshow("raw", img);

	Mat bImage;	
	medianBlur(img, img, 3);
	imshow("blur", img);
	adaptiveThreshold(img, bImage, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 3, 10);
	imshow("binary", bImage);
	//imwrite("binary.jpg", bImage);

	//log_normalization(img);
	Mat elem(4, 4, CV_8U, Scalar(1));
	//morphologyEx(bImage, bImage, MORPH_CLOSE, elem);
	dilate(bImage, bImage, elem);
	
	imshow("connection", bImage);
	//imwrite("connection.jpg", bImage);
	//imwrite("results/binary2.png", bImage);
	
	vector<vector<Point> > contours; // Vector for storing contours
	vector<Vec4i> hierarchy;

    findContours(bImage, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE ); // Find the contours in the image
	cout << "contour number: " << contours.size() << endl;
	
	Mat drawing = Mat::zeros(bImage.size(), CV_8UC3);

	int num = 0;

    for(auto it = contours.begin(); it != contours.end(); it++){ // iterate through each contour.
        double area = contourArea(*it);  
		//  Find the area of contour
		if(area > 10){
			cout << area << endl;
			num ++;
		}
		else{
			contours.erase(it);	
		}
		
    }
	

	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());

	int count = 0;
 	for (int i = 0; i < contours.size(); i++){
		minRect[i] = minAreaRect(Mat(contours[i]));
		if (contours[i].size() > 5){
			minEllipse[i] = fitEllipse(Mat(contours[i]));
			if(minEllipse[i].angle > 110 && minEllipse[i].angle < 150)
				count += 1;
		}
	} 	
	
	cout << "the number of yulin: " << count << endl;
	/// 绘出轮廓及其可倾斜的边界框和边界椭圆	

	for (int i = 0; i < contours.size(); i++){
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// contour		
		//drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		// ellipse
		ellipse(drawing, minEllipse[i], color, 2, 8);
		// rotated rectangle
		Point2f rect_points[4]; minRect[i].points(rect_points);
		//for (int j = 0; j < 4; j++)
			//line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
	}
 	/// 结果在窗体中显示
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	//imwrite("drawing.jpg", drawing);
	waitKey();
	
	return 0;
}
