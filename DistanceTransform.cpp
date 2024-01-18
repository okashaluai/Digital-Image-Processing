#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include <limits>

using namespace cv; 
using namespace std;





//Complexity - O(nm)
void distance_transform(const Mat &img, Mat &distance_transform){

    //initialize
    vector<vector<Point>> contours;
    vector<Vec4i> _;
    Point p;

    findContours(img, contours, _, RETR_EXTERNAL, CHAIN_APPROX_NONE );
    distance_transform = cv::Mat(img.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));

    for( int i = 0; i < contours.size(); i++){
            for(int j = 0; j < contours[i].size(); j++){
                p = contours[i][j];
                distance_transform.at<float>(p.y, p.x) = 0.0;
            }
    }
    
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.at<uchar>(y, x) == 0) {
                distance_transform.at<float>(y, x) = 0.0;
            }
        }
    }

    
    //forward scan
    for( int y = 0; y < distance_transform.rows; y++){
        for( int x = 0; x < distance_transform.cols; x++){
            for( int dy = -2; dy <= 2; dy++){
                for( int dx = -2; dx <= 2; dx++){
                    if(y + dy >= 0 && y + dy < distance_transform.rows && x + dx >= 0 && x + dx < distance_transform.cols){
                        distance_transform.at<float>(y + dy, x + dx) = std::min(distance_transform.at<float>(y + dy, x + dx), distance_transform.at<float>(y, x) + sqrt(static_cast<float>(dx * dx + dy * dy)));
                    }
                }
            }
            }
        }
    



    //backward scan
    for( int y = distance_transform.rows - 1; y >= 0; y--){
        for( int x = distance_transform.cols - 1; x >= 0; x--){
            for( int dy = -2; dy <= 2; dy++){
                for( int dx = -2; dx <= 2; dx++){
                    if(y + dy >= 0 && y + dy < distance_transform.rows && x + dx >= 0 && x + dx < distance_transform.cols){
                        distance_transform.at<float>(y + dy, x + dx) = std::min(distance_transform.at<float>(y + dy, x + dx), distance_transform.at<float>(y, x) + sqrt(static_cast<float>(dx * dx + dy * dy)));
                    }
                }
            }
        }
    }
}


       

//for testing
void printImageMatrix(Mat img){
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            cout << img.row(i).col(j)<< " ";
        }
        cout << endl;
    }
}



void component_threshold(Mat &src_img, unsigned int threshold_size){
        Mat labels, stats, centroids;

        int numOfComponents = connectedComponentsWithStats(src_img, labels, stats, centroids, 8, CV_16U);

        // [0] is the background component.
        for (int i = 1; i < numOfComponents; i++) {
            if (stats.at<int>(i, cv::CC_STAT_HEIGHT) < threshold_size || stats.at<int>(i, cv::CC_STAT_WIDTH) < threshold_size){
                for(int y = stats.at<int>(i, CC_STAT_TOP); y < stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT);y++){
                    for (int x = stats.at<int>(i, CC_STAT_LEFT); x < stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH);x++){
                        src_img.at<uchar>(y, x) = 0;
                    }
                }                
            }
        }

}

//saving an image to file
void save_image(const Mat output_image, const char *output_path){
    Mat product;
    if(output_path){
        normalize(output_image, product, 0, 255, NORM_MINMAX, CV_8U);;
        imwrite(output_path, product);
    }
}


int main(int argc, char** argv){
    
    //args handling
    if(argc < 4){
        cout << "<image path> <component size threshold> <I/O/S/All> <Optional output path>" << endl; 
        exit(1);
    }

    char *input_path = argv[1];
    unsigned int component_thresh = atoi(argv[2]);
    char type = argv[3][0];
    char *output_path = 0;

    if (argc == 5)output_path = argv[4];

    //grayscalling 
    Mat original_image = imread(input_path, IMREAD_COLOR);
    Mat gray_image;
    cvtColor(original_image, gray_image, COLOR_BGR2GRAY, 1);

    //binarization
    Mat binary_image;
    threshold(gray_image, binary_image, 128, 255, THRESH_BINARY); // 0, ...128.. , 255: 0 is black & 255 is white, 
    
    //components cleaning
    component_threshold(binary_image, component_thresh);

    //DT calculating
    if(type == 'i'){
        Mat inner_distance_transform;
        distance_transform(binary_image, inner_distance_transform);
        cv::normalize(inner_distance_transform, inner_distance_transform, 0, 1, cv::NORM_MINMAX);
        imshow("Binary Image", binary_image);
        imshow("Inner Distance Transform", inner_distance_transform);
        save_image(inner_distance_transform, output_path);
    }
    else if(type == 'o'){
        Mat outer_distance_transform;
        distance_transform(255 - binary_image, outer_distance_transform);
        cv::normalize(outer_distance_transform, outer_distance_transform, 0, 1, cv::NORM_MINMAX);
        
        imshow("Binary Image", binary_image);
        imshow("Outer Distance Transform", outer_distance_transform);
        save_image(outer_distance_transform, output_path);
    }
    else if(type == 's'){
        Mat outer_distance_transform;
        distance_transform(255 - binary_image, outer_distance_transform);
        
        Mat inner_distance_transform;
        distance_transform(binary_image, inner_distance_transform);

        Mat signed_distance_transform = outer_distance_transform - inner_distance_transform;
        cv::normalize(signed_distance_transform, signed_distance_transform, 0, 1, cv::NORM_MINMAX);
        
        imshow("Binary Image", binary_image);
        imshow("Signed Distance Transform", signed_distance_transform);
        save_image(signed_distance_transform, output_path);
    }
    else{
        Mat outer_distance_transform;
        distance_transform(255 - binary_image, outer_distance_transform);
        
        Mat inner_distance_transform;
        distance_transform(binary_image, inner_distance_transform);

        Mat signed_distance_transform = outer_distance_transform - inner_distance_transform;

        cv::normalize(outer_distance_transform, outer_distance_transform, 0, 1, cv::NORM_MINMAX);
        cv::normalize(inner_distance_transform, inner_distance_transform, 0, 1, cv::NORM_MINMAX);
        cv::normalize(signed_distance_transform, signed_distance_transform, 0, 1, cv::NORM_MINMAX);
        
        imshow("Binary Image", binary_image);
        imshow("Outer Distance Transform", outer_distance_transform);
        imshow("Inner Distance Transform", inner_distance_transform);
        imshow("Signed Distance Transform", signed_distance_transform);
    }

    moveWindow("Binary Image", 10, 50);
    moveWindow("Outer Distance Transform", 650, 50); 
    moveWindow("Inner Distance Transform", 10, 500);
    moveWindow("Signed Distance Transform", 650, 500); 

    waitKey(0);
    destroyAllWindows();

    exit(0);
}