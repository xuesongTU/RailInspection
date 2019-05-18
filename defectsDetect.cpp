#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <vector>  
#include <algorithm>

using namespace std;
using namespace cv;

//adaptive Median Filter

uchar adaptiveProcess(const Mat &im, int row,int col,int kernelSize,int maxSize)
{
    vector<uchar> pixels;
    for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
        for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
        {
            pixels.push_back(im.at<uchar>(row + a, col + b));
        }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = im.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to B
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveProcess(im, row, col, kernelSize, maxSize);
 //增大窗口尺寸，继续A过程。
        else
            return med;
    }
}

int main(int argc, char* argv[]){
	Mat img = imread("/home/xuesong/RailInspection/defectsImages/yulinshang/y6.png");
	cout << "hello" << img.channels() << endl;
	
	double t0 = (double)getTickCount();
	if (img.empty())
		return -1;
	if (img.channels() != 1)
	{
		cvtColor(img, img, CV_BGR2GRAY);
	}
	if (img.depth() == CV_8U)
	{
		img.convertTo(img, CV_32F);
	}

	int minSize = 5; // 滤波器窗口的起始尺寸
    int maxSize = 7; // 滤波器窗口的最大尺寸
    Mat im1;
    // 扩展图像的边界
    copyMakeBorder(img, im1, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BorderTypes::BORDER_REFLECT);
    // 图像循环
    for (int j = maxSize / 2; j < im1.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < im1.cols * im1.channels() - maxSize / 2; i++)
        {
            im1.at<uchar>(j, i) = adaptiveProcess(im1, j, i, minSize, maxSize);
        }
    }
	
	imshow("median_filter", im1);
	
	//double t0 = (double)getTickCount();
	//top-hat operation
	//cv::Mat element(4, 4, CV_8U, cv::Scalar(1));
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	cv::morphologyEx(img, img, cv::MORPH_TOPHAT, element);
	t0 = (double)getTickCount() - t0;
	cout << " time:\n " << t0 * 1000 / getTickFrequency() << " ms" << endl;
	imshow("tophat", img);
	waitKey();


	return 0;
}
