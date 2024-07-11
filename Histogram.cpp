#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace std;

enum MODE {HISTOGRAM_EQUALIZATION = 'e', HISTOGRAM_MATCHING = 'm', HISTOGRAM_DRAWING = 'h'};


void calculate_histogram(cv::Mat &img, vector<int> & histogram){

    for(int y = 0; y < img.rows; y++){
        for(int x = 0; x < img.cols; x++){
            histogram[img.at<uchar>(y, x)]++;
        
        }

    }

}


void normalize_histogram(vector<int> & histogram , vector<double> & normalized_histogram , const  double height, const  double width){
    

    for(int i = 0; i < 256; i++){
        normalized_histogram[i] = histogram[i]/(height * width);
    }

    
}

void calculate_cdf(vector<int> & histogram, vector<double> & cdf, const double height, const double width){
    vector<double> normalized_histogram(256, 0);
    normalize_histogram(histogram, normalized_histogram, height, width);

    cdf[0] = normalized_histogram[0];

    for(int i = 1 ; i < 256; i++){
        cdf[i] = cdf[i - 1] + normalized_histogram[i];
    }


}

void equalize_histogram(cv::Mat &img,vector<int> & histogram, cv::Mat &equalized){
    
    vector<double> cdf(256,0);

    calculate_cdf(histogram, cdf, img.rows, img.cols);
    
    for(int y = 0; y < equalized.rows; y++){
        for(int x = 0; x < equalized.cols; x++){

            equalized.at<uchar>(y, x) = static_cast<uchar>(255 * cdf[img.at<uchar>(y,x)]);

        }

    }

}
void match_histogram(cv::Mat &source, cv::Mat &reference, cv::Mat& matched){

            vector<int> source_histogram(256);
            vector<double> source_histogram_cdf(256);
            vector<int> reference_histogram(256);
            vector<double> reference_histogram_cdf(256);
            vector<int> lookup_table(256);

            calculate_histogram(source, source_histogram);
            calculate_cdf(source_histogram, source_histogram_cdf, source.rows, source.cols);

            calculate_histogram(reference, reference_histogram);
            calculate_cdf(reference_histogram, reference_histogram_cdf, reference.rows, reference.cols);


            int min_diff_color;
            

            for(int i = 0; i < 256; i++){
                
                min_diff_color = 0;

                for(int j = 0; j < 256; j++){
                    if (abs(reference_histogram_cdf[j] - source_histogram_cdf[i]) < abs(reference_histogram_cdf[min_diff_color] - source_histogram_cdf[i])){
                        min_diff_color = j;
                    }
                }

                lookup_table[i] = min_diff_color;
                
            }

            for(int y = 0; y < matched.rows; y++){
                for(int x = 0; x < matched.cols; x++){
                    matched.at<uchar>(y,x) = lookup_table[matched.at<uchar>(y,x)];
                }
            }
                        
}


void display_histogram(const char * title, const vector<double> & histogram, int pixels_num){
    pid_t child_pid = fork();
    if (child_pid == 0){
        int start = 0;
        int end = 255;

        int size = end - start + 1;
        std::vector<int> x(size );
        std::iota(x.begin(), x.end(), start);
        
        plt::plot(x, histogram);
        plt::xlabel("r");
        plt::ylabel("P(r)");
        plt::title(title);
        plt::legend();
        plt::show();

        _exit(EXIT_SUCCESS);

    }


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
    //args handling
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode: H/E/M> <input_image_path> [<match_image_path>] [<output_image_path>]" << std::endl;
        return -1;
    }

    char mode = cv::toLowerCase(argv[1])[0];
    char *source_path = argv[2];
    char *output_path = 0;
    char * reference_path = 0;
 
    cv::Mat source_image = cv::imread(source_path, cv::IMREAD_COLOR);
    
    unsigned int height = source_image.rows;
    unsigned int width = source_image.cols;
    cv::Mat source_gray_image;

    cv::cvtColor(source_image, source_gray_image, cv::COLOR_BGR2GRAY);


    if(mode == HISTOGRAM_DRAWING){
        
        std::vector<int> histogram(256, 0);
        std::vector<double> normalized_histogram(256, 0);

        calculate_histogram(source_gray_image, histogram);
        normalize_histogram(histogram, normalized_histogram, height, width);
        cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Source Image", source_gray_image);

        display_histogram("Histogram", normalized_histogram, height * width);



    }

    else if(mode == HISTOGRAM_EQUALIZATION){

        if(argc == 4){
            output_path = argv[3];
        }

        std::vector<int> histogram(256, 0);
        cv::Mat histogram_equalized_image = source_gray_image.clone();

        calculate_histogram(source_gray_image, histogram);     
        equalize_histogram(source_gray_image, histogram ,histogram_equalized_image);
        cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);

        cv::namedWindow("Histogram Equalized Image", cv::WINDOW_AUTOSIZE);

        cv::imshow("Source Image", source_gray_image);

        cv::imshow("Histogram Equalized Image", histogram_equalized_image);

        std::vector<int> result_histogram(256, 0);
        std::vector<double> normalized_result_histogram(256, 0);
        std::vector<double> normalized_histogram(256, 0);
        calculate_histogram(histogram_equalized_image, result_histogram);
        normalize_histogram(result_histogram, normalized_result_histogram, height, width);
        normalize_histogram(histogram, normalized_histogram, height, width);

        display_histogram("Source Histogram",normalized_histogram, height * width);
        display_histogram("Equalization Histogram", normalized_result_histogram, height * width);

        save_image(histogram_equalized_image, output_path);


    }

    else if(mode== HISTOGRAM_MATCHING){

        if(argc == 4){
            reference_path = argv[3];
        }
        if(argc == 6){
            reference_path = argv[3];
            output_path = argv[5];
        }
        cv::Mat reference_image = cv::imread(reference_path, cv::IMREAD_COLOR);
        cv::Mat reference_gray_image;
        cv::Mat matched_image = source_gray_image.clone();

        cvtColor(reference_image, reference_gray_image, cv::COLOR_BGR2GRAY);

        match_histogram(source_gray_image, reference_gray_image, matched_image);
        cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Reference Image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Matched Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Source Image", source_gray_image);
        cv::imshow("Reference Image", reference_gray_image);
        cv::imshow("Matched Image", matched_image);
        
        vector<int> source_histogram(256, 0), reference_histogram(256, 0), matched_histogram(256, 0); 
        vector<double> normalized_source_histogram(256, 0), normalized_reference_histogram(256, 0), normalized_matched_histogram(256, 0); 


        calculate_histogram(source_gray_image, source_histogram);
        calculate_histogram(reference_gray_image, reference_histogram);
        calculate_histogram(matched_image, matched_histogram);

        normalize_histogram(source_histogram, normalized_source_histogram, source_gray_image.rows, source_gray_image.cols);
        normalize_histogram(reference_histogram, normalized_reference_histogram, reference_gray_image.rows, reference_gray_image.cols);
        normalize_histogram(matched_histogram, normalized_matched_histogram, matched_image.rows, matched_image.cols);
        
        display_histogram("Source Histogram", normalized_source_histogram, source_gray_image.rows * source_gray_image.cols);
        display_histogram("Reference Histogram", normalized_reference_histogram, reference_gray_image.rows * reference_gray_image.cols);
        display_histogram("Matched Histogram", normalized_matched_histogram, matched_image.rows * matched_image.cols);

        save_image(matched_image, output_path);
    }
    else{            
        std::cerr << "Invalid mode." << std::endl;
        return -1;
    }    

    cv::waitKey(0);
    cv::destroyAllWindows();

    exit(0);
}
