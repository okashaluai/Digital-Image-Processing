#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include <cassert>
#include <math.h>

using namespace std;


bool hit(const cv::Mat & image, int y, int x, const cv::Mat & structure_element){

    int structure_element_h = structure_element.rows;
    int structure_element_w = structure_element.cols;

    for(int dy = 0; dy < structure_element_h; dy++){
        for(int dx = 0; dx < structure_element_w; dx++){
            if (!(y + dy < image.rows && (y + dy >= 0) && x + dx < image.cols && (x + dx >= 0) )) return false;
            if(structure_element.at<uchar>(dy,dx) ==  image.at<uchar>(y + dy, x + dx)) continue;
            else return false;
            
        }
    }

    return true;
}

bool fit(const cv::Mat & image, int y, int x, const cv::Mat & structure_element){

    int structure_element_h = structure_element.rows;
    int structure_element_w = structure_element.cols;

    for(int dy = 0; dy < structure_element_h  ; dy++){
        for(int dx = 0; dx < structure_element_w; dx++){
            if (structure_element.at<uchar>(dy, dx) == 0) continue;

            int image_y = y + dy - structure_element_h + 1;
            int image_x = x + dx - structure_element_w + 1;

            if (!(image_y < image.rows && (image_y >= 0) && image_x < image.cols && (image_x >= 0) )) continue;;
            if (image.at<uchar>(image_y, image_x) ==  structure_element.at<uchar>(dy, dx)){
                return true;
            }
        }
    }

    return false;
}

void erosion(const cv::Mat & input_image, cv::Mat & output_image, const cv::Mat & structure_element){
    for(int y = 0; y < output_image.rows; y++){
        for(int x = 0; x < output_image.cols; x++){
            if (hit(input_image, y, x, structure_element)){
                output_image.at<uchar>(y, x) = 255;
            }else{
                output_image.at<uchar>(y, x) = 0;
            }
        }
    }
}
void dilation(const cv::Mat input_image, cv::Mat output_image, const cv::Mat & structure_element){
    for(int y = 0; y < output_image.rows; y++){
        for(int x = 0; x < output_image.cols; x++){
            if(input_image.at<uchar>(y, x) == 255) continue;
            if (fit(input_image, y, x, structure_element)){
                output_image.at<uchar>(y, x) = 255;
            }else{
                output_image.at<uchar>(y, x) = 0;
            }

        }
    }
}


 int getMedianArea(const cv::Mat &binaryImage, const cv::Mat & stats, int labelsN){
    vector<int> areas;
    for(int i = 1; i < labelsN; i++){
        areas.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
    }
    sort(areas.begin(), areas.end());

    // areas.erase(std::unique(areas.begin(), areas.end()), areas.end());

    int median_area = areas[(areas.size()/2) + 1];
    return median_area;
}

vector<cv::Mat> getUniqueStructureElements(vector<cv::Mat> SEs){
    
    std::sort(SEs.begin(), SEs.end(), [](const cv::Mat& a, const cv::Mat& b) {
        return cv::countNonZero(a) < cv::countNonZero(b);
    });

    auto it = std::unique(SEs.begin(), SEs.end(), [](const cv::Mat& a, const cv::Mat& b) {
        if (a.size() != b.size()) {
            return false;   
        }
        cv::Mat c = cv::Mat::zeros(a.size(), a.type());
        cv::bitwise_xor(a, b, c);        

        return cv::countNonZero(c) == 0; 
    });

    SEs.erase(it, SEs.end());
    return SEs;
}


vector<cv::Mat> getStructureElements(const cv::Mat &binaryImage, const cv::Mat & stats, int labelsN){
        
    int medianCharacterArea = getMedianArea(binaryImage, stats, labelsN);
    vector<cv::Mat> SEs;
    for(int i = 1; i < labelsN; i++){
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > medianCharacterArea) continue;

        cv::Mat SE = cv::Mat::zeros(cv::Size(width + 2, height + 2), binaryImage.type());

        for(int y = 1; y < SE.rows - 1; y++){
            for(int x = 1; x < SE.cols - 1; x++){
                SE.at<uchar>(y, x) = binaryImage.at<uchar>(stats.at<int>(i, cv::CC_STAT_TOP) + y - 1, stats.at<int>(i, cv::CC_STAT_LEFT) + x - 1);
            }
        }
        SEs.push_back(SE);
    }

    return SEs;
}

void unDemarcate(const cv::Mat &Text, cv::Mat &product){
    cv::Mat labels, stats, centroids;
    int labelsN = cv::connectedComponentsWithStats(Text, labels, stats, centroids);
    

    vector<cv::Mat> SEs = getUniqueStructureElements(getStructureElements(Text, stats, labelsN));

    cv::Mat Niqquds = cv::Mat::zeros(Text.size(), Text.type());


    for(int i = 0; i < SEs.size(); i++){

        cv::Mat SE = SEs[i];
        cv::Mat tempErosion = cv::Mat::zeros(Text.size(), Text.type());
        cv::Mat tempDilation = cv::Mat::zeros(Text.size(), Text.type());
        erosion(Text, tempErosion, SE);
        cv::rotate(SE, SE, cv::ROTATE_180);
        dilation(tempErosion, tempDilation ,SE);
        Niqquds |= tempDilation;
    }

    cv::imshow("Niqquds", ~Niqquds);
    cv::imshow("Original Text", ~Text);
    product = (Text) - Niqquds;
    cv::imshow("Product Text",~product);
    cv::waitKey(0);
    cv::destroyAllWindows();

}
//saving an image to file
void save_image(const cv::Mat output_image, const char *output_path){
    cv::Mat product;
    if(output_path){
        normalize(output_image, product, 0, 255, cv::NORM_MINMAX, CV_8U);;
        imwrite(output_path, product);
    }
}

 
int main(int argc, char** argv){
    char *output_path = 0;
    if (argc == 3)output_path = argv[2];

    cv::Mat source_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat source_gray_image;
    cv::cvtColor(source_image, source_gray_image, cv::COLOR_BGR2GRAY);
    cv::Mat Text; 
    cv::threshold(source_gray_image, Text, 127, 255, cv::THRESH_BINARY_INV);
    cv::Mat Product = cv::Mat::zeros(Text.size(), Text.type());
    unDemarcate(Text, Product);
    save_image(Product, output_path);
    return 0;

}