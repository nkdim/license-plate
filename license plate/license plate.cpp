#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp" 
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stack>  
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <cstring> 
#include <opencv2/ml/ml.hpp>
#include <sstream>
#include <algorithm>


using namespace std;
using namespace cv;
using namespace cv::ml;

namespace cv
{
	using std::vector;
}

void imageits_preprocessing(Mat&, Mat&);
void imagecolor(Mat&, Mat&);

const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 30;
const int RESIZED_IMAGE_HEIGHT = 60;

vector<char> digits;



class myclass {
public:
	myclass(int a, int b, int c, int d) :first(a), second(b), third(c), fourth(d) {}
	int first;
	int second, third, fourth;
	bool operator < (const myclass &m)const {
		return first < m.first;
	}
};

class ContourWithData {
public:
	// member variables ///////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point> ptContour;          
	cv::Rect boundingRect;                   
	float fltArea;                            

												///////////////////////////////////////////////////////////////////////////////////////////////
	bool checkIfContourIsValid() {                            
		if (fltArea < MIN_CONTOUR_AREA) return false;       
		return true;                                           
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   
	}

};


void knn(Mat & img)
{

	std::vector<ContourWithData> allContoursWithData;          
	std::vector<ContourWithData> validContoursWithData;        

																 ///////////////////////////////////////////////////

	cv::Mat matClassificationInts;      

	cv::FileStorage fsClassifications("license plate_classifications.xml", cv::FileStorage::READ);      



	fsClassifications["classifications"] >> matClassificationInts;      
	fsClassifications.release();                                       

																		////////////////////////////////////////////////////////////

	cv::Mat matTrainingImagesAsFlattenedFloats;        

	cv::FileStorage fsTrainingImages("license plate_new_images.xml", cv::FileStorage::READ);         



	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;          
	fsTrainingImages.release();                                                 

																				//////////////////////////////////////////////////////////////////////////////

	cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());          

																				
																				
	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);



	std::string strFinalString;
	cv::Mat matROIResized;

	cv::resize(img, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));    

	cv::Mat matROIFloat;
	matROIResized.convertTo(matROIFloat, CV_32FC1);            

	cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

	cv::Mat matCurrentChar(0, 0, CV_32F);

	kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);    

	float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

	strFinalString = strFinalString + char(int(fltCurrentChar));      


																	
	char *cstr = &strFinalString[0u];

	digits.push_back(*cstr);


}


void imagecolor(Mat& roi, Mat& image)
{
	int roi_rows = (roi.rows*0.4);

	//cout << "roi_rows = " << roi_rows << std::endl;

	vector< myclass > vect;

	cv::Mat labelImage;
	cv::Mat stats, centroids;
	Mat cropedImage = roi;

	int nLabels = cv::connectedComponentsWithStats(cropedImage, labelImage, stats, centroids, 8, CV_32S);/////八連通
	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);
	//std::cout << "Number of connected components = " << nLabels << std::endl << std::endl;
	int lab = 0;

	for (int label = 1; label < nLabels; ++label) {
		colors[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
		int labh = stats.at<int>(label, cv::CC_STAT_HEIGHT);
		int labw = stats.at<int>(label, cv::CC_STAT_WIDTH);



		/*cout << "labh = " << stats.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;
		cout << "labw = " << stats.at<int>(label, cv::CC_STAT_WIDTH) << std::endl;*/

		//cout << std::endl;
		if ((stats.at<int>(label, cv::CC_STAT_HEIGHT)< stats.at<int>(label, cv::CC_STAT_WIDTH)) || (stats.at<int>(label, cv::CC_STAT_HEIGHT)<roi_rows))
		{
			colors[label] = cv::Vec3b(0, 0, 0);

		}
		else
		{
			colors[label] = cv::Vec3b(255, 255, 255);
			//cout << "labh = " << stats.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;
			//cout << "labw = " << stats.at<int>(label, cv::CC_STAT_WIDTH) << std::endl;
			lab += 1;

			myclass my(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP), stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);
		}
	}

	sort(vect.begin(), vect.end());//排序位置

	cv::Mat cc(roi.size(), CV_8UC3);////上色

	//cout << "字元數量: " << lab << std::endl;

	for (int r = 0; r < cc.rows; ++r) {
		for (int c = 0; c < cc.cols; ++c) {
			int label = labelImage.at<int>(r, c);
			cv::Vec3b &pixel = cc.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}

	cvtColor(cc, cc, cv::COLOR_BGR2GRAY);

	threshold(cc, cc, 150, 255, THRESH_BINARY | THRESH_OTSU);


	if (lab == 7)
	{

		Mat d1 = cc(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = cc(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = cc(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = cc(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = cc(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = cc(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = cc(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));

		cv::resize(d1, d1, cv::Size(30, 60));
		cv::resize(d2, d2, cv::Size(30, 60));
		cv::resize(d3, d3, cv::Size(30, 60));
		cv::resize(d4, d4, cv::Size(30, 60));
		cv::resize(d5, d5, cv::Size(30, 60));
		cv::resize(d6, d6, cv::Size(30, 60));
		cv::resize(d7, d7, cv::Size(30, 60));



		imshow("d1", d1);
		imwrite("d1.jpg", d1);


		knn(d1);
		knn(d2);
		knn(d3);
		knn(d4);
		knn(d5);
		knn(d6);
		knn(d7);

	}

	imshow("imagecolor", cc);

	cout << "車牌辨識結果: ";

	vector<char>  ::iterator iter = digits.begin();
	for (int ix = 0; iter != digits.end(); ++iter, ++ix) {


		cout << *iter;





	}




}


void imageits_preprocessing(Mat& roi, Mat& image)
{


	cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
	GaussianBlur(roi, roi, Size(3, 3), 0, 0);
	threshold(roi, roi, 150, 255, THRESH_BINARY | THRESH_OTSU);

	bitwise_not(roi, roi);

	imshow("roi_img", roi);

	imagecolor(roi, image);


}

int main(int argc, char** argv)
{


	Mat src = imread("L1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat img = imread("L1.jpg");

	Mat image = imread("L1.jpg");

	Mat dst, grad_x, His;

	GaussianBlur(src, src, Size(5, 5), 0, 0);

	equalizeHist(src, dst);


	int iterations = 15;



	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));//CLOSE
	morphologyEx(dst, grad_x, MORPH_OPEN, kernel2, cv::Point(-1, -1), iterations);

	imshow("dst", grad_x);

	Mat sub, can;

	subtract(dst, grad_x, sub);

	threshold(sub, sub, 150, 255, THRESH_BINARY | THRESH_OTSU);

	Canny(sub, can, 50, 150, 3);


	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(can, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	Scalar color(0, 0, 255);

	for (int i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;
			bounding_rect = boundingRect(contours[i]);
		}

	}


	Scalar color2(255, 0, 0);

	rectangle(img, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);

	Mat roi = image(bounding_rect).clone();


	imageits_preprocessing(roi, image);

	imshow("img", img);

	//imshow("grad_x", sub);

	//imshow("can", can);

	imshow("roi", roi);
	waitKey(0);

	return 0;
}