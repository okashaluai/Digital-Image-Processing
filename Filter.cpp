#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include <cassert>
#include <math.h>

using namespace std;

enum FilterMODE {MIN , MAX ,  MEDIAN };

vector<vector<double>> getSobelBySize(int sobelSize){
    vector<vector<double>> sobelY;
    assert(sobelSize % 2 == 1);

    if (sobelSize == 3){
        sobelY = {{1, 2, 1}, 
                  {0, 0, 0}, 
                  {-1, -2, -1}};
    }else if (sobelSize == 5){
        sobelY = {{2, 2, 4, 2, 2}, 
                  {1, 1, 2, 1, 1},
                  {0, 0, 0, 0, 0}, 
                  {-1, -1, -2, -1, -1}, 
                  {-2, -2, -4, -2, -2}};
    }else if(sobelSize == 7){
        sobelY = {{3, 4, 5, 6, 5, 4, 3}, 
                  {2, 3, 4, 5, 4, 3, 2},
                  {1, 2, 3 ,4, 3, 2, 1},
                  {0, 0, 0, 0, 0, 0, 0}, 
                  {-1, -2, -3 ,-4, -3, -2, -1}, 
                  {-2, -3, -4, -5, -4, -3, -2}, 
                  {-3, -4, -5, -6, -5, -4, -3}};
    }else if (sobelSize == 9){
        sobelY = {{4, 5, 6, 7, 8, 7, 6, 5, 4},
                  {3, 4, 5, 6, 7, 6, 5, 4, 3},
                  {2, 3, 4, 5, 6, 5, 4, 3, 2},
                  {1, 2, 3, 4, 5, 4, 3, 2, 1},
                  {0, 0, 0, 0, 0, 0, 0, 0, 0},
                  {-1, -2, -3, -4, -5, -4, -3, -2, -1},
                  {-2, -3, -4, -5, -6, -5, -4, -3, -2},
                  {-3, -4, -5, -6, -7, -6, -5, -4, -3},
                  {-4, -5, -6, -7, -8, -7, -6, -5, -4}};
    }else{
        cout << "Sobel of size " << sobelSize << " is not supported!" << endl;
    }

    return sobelY;   
}


vector<vector<double>> transpseSobel(vector<vector<double>> &Gy){
    vector<vector<double>> Gx(Gy.size(), vector<double>(Gy.size(), 0));
    double sum = 0.0;
    for(int i = 0; i < Gy.size(); i++){
        for(int j = 0; j < Gy.size(); j++){
            Gx[i][j] = Gy[j][i];
            sum += abs(Gx[i][j]);
        }
    }
    return Gx;
}

void normalizeKernel(vector<vector<double>> & kernel){
    double sum = 0.0;

    for(int i = 0; i < kernel.size(); i++){
        for(int j = 0; j < kernel.size(); j++){
            sum += abs(kernel[i][j]);
        }
    }
    for(int i = 0; i < kernel.size(); i++){
        for(int j = 0; j < kernel.size(); j++){
            kernel[i][j] /= sum;
        }
    }
}


void Padding(const cv::Mat & image, cv::Mat & padded, int n){
    int new_width = n + image.cols + n;
    int new_height = n + image.rows + n;
    padded = cv::Mat::zeros(cv::Size(new_width, new_height), image.type());
    
    for(int y  = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            padded.at<uchar>(y + n,x + n) = image.at<uchar>(y,x);
        }
    }

    //upper padding
    for(int y  = 0; y < n ; y++){
        for(int x = n; x < padded.cols - n; x++){
            padded.at<uchar>(y,x) = padded.at<uchar>(2 * n - y, x);
        }
    }

    //lower padding
    for(int y  = (padded.rows - n - 1); y < padded.rows ; y++){
        for(int x = n; x < padded.cols - n; x++){
            padded.at<uchar>(y,x) = padded.at<uchar>( 2 * (padded.rows - n - 1) - y , x);
        }
    }

    //right padding
    for(int y  = 0; y < padded.rows ; y++){
        for(int x = (padded.cols - n - 1); x < padded.cols; x++){
            padded.at<uchar>(y,x) = padded.at<uchar>(y, 2 * (padded.cols - n - 1) - x);
        }
    }

    //left padding
    for(int y  = 0; y < padded.rows ; y++){
        for(int x = 0; x < n; x++){
            padded.at<uchar>(y,x) = padded.at<uchar>(y, 2 * n - x);
        }
    }
}

void applyFilter(const cv::Mat &image, cv::Mat & filtered_image,vector<vector<double>> filterKernel, int kernelSize){
    cv::Mat padded;
    Padding(image, padded, kernelSize/2);
    
    double new_value;
    for(int y = 0; y < filtered_image.rows; y++){
        for(int x = 0; x < filtered_image.cols; x++){
            new_value = 0.0;
            for(int dy = -(kernelSize/2); dy <= (kernelSize/2); dy++){
                for(int dx = -(kernelSize/2); dx <= (kernelSize/2); dx++){
                    new_value += (padded.at<uchar>(y + (kernelSize/2) + dy ,  x + (kernelSize/2) + dx ) * (filterKernel[(kernelSize/2) + dy][(kernelSize/2) + dx]));
                }
            }
            filtered_image.at<uchar>(y,x) = static_cast<uchar>(new_value);
        }
    }    
}



void GaussianFilter(const cv::Mat &image, cv::Mat &filter_response, int kernelSize, double sigma){
    auto gaussian = [innersigma = sigma](int x, int y) -> double {
        return (1/(2 * M_PI * pow(innersigma, 2))) * exp(-(pow(x, 2) + pow(y, 2))/(2 * pow(innersigma, 2)));
    };

    auto generate_gaussian_kernel = [kernelSize = kernelSize, sigma = sigma](auto gaussian) -> vector<vector<double>> {
        vector<vector<double>> filterKernel(kernelSize, vector<double>(kernelSize, 0.0));

        double sum = 0.0;
        for(int x = 0; x < kernelSize; x++){

            for(int y = 0; y < kernelSize; y++){
                filterKernel[x][y] = gaussian(x - (kernelSize/2),y - (kernelSize/2));
                sum += filterKernel[x][y];
            }
        }

        cout << "---------------------------Gaussian Filter---------------------------" << endl;
        cout << "Sigma: " << sigma << " , " << "Filter size: " << kernelSize << endl;
        for(int i = 0; i < kernelSize; i++){
            for(int j = 0; j < kernelSize; j++){
                filterKernel[i][j] /= sum;
                cout << filterKernel[i][j] << " ";
            }
            cout << endl;
        }
        cout << "---------------------------------------------------------------------" << endl;

        return filterKernel;
    };

    vector<vector<double>> filter_kernel = generate_gaussian_kernel(gaussian);

    applyFilter(image, filter_response, filter_kernel, kernelSize);
}

void edgeDetection(const cv::Mat &image, cv::Mat &edges, int sobelSize, int threshold, int yd, int xd){
    cv::Mat blurred = cv::Mat::zeros(image.size(), image.type());
    cv::Mat padded;

    GaussianFilter(image, blurred, 3, 1.0);
    Padding(blurred, padded, sobelSize/2);

    vector<vector<double>> kernelY = getSobelBySize(sobelSize);
    normalizeKernel(kernelY);
    vector<vector<double>> kernelX = transpseSobel(kernelY);

    double Gx, Gy, G;

    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            
            Gx = 0, Gy = 0;

            for(int dy = -(sobelSize/2); dy <= (sobelSize/2); dy++){
                for(int dx = -(sobelSize/2); dx <= (sobelSize/2); dx++){
                    if (xd)
                        Gx += (padded.at<uchar>(y + (sobelSize/2) + dy ,  x + (sobelSize/2) + dx ) * (kernelX[(sobelSize/2) + dy][(sobelSize/2) + dx]));
                    if(yd)
                        Gy += (padded.at<uchar>(y + (sobelSize/2) + dy ,  x + (sobelSize/2) + dx ) * (kernelY[(sobelSize/2) + dy][(sobelSize/2) + dx]));

                }
            }
            if (xd == 1 && yd == 1)
                G =  min(255.0, sqrt((Gx * Gx) + (Gy * Gy )));
            else if (xd == 1)
                G = min(255.0, sqrt(Gx * Gx));
            else if (yd == 1)
                G = min(255.0, sqrt(Gy * Gy));
            else {}
            edges.at<uchar>(y, x) = static_cast<uchar>(G);
        }
    }
    cv::normalize(edges, edges, 0, 255, cv::NORM_MINMAX);
    cv::threshold(edges, edges, threshold, 255, cv::THRESH_BINARY);
}    

void SelectionFilter(cv::Mat & input_image, cv::Mat & output_image, int size, FilterMODE filterMode){

    assert(size%2 != 0);
    cv::Mat padded;
    Padding(input_image, padded, (size/2));
    vector<uchar> buffer(0);


    switch (filterMode)
    {
    case MIN:
        uchar current_min;
        
        for(int y = 0; y < input_image.rows; y++){
            for(int x = 0; x < input_image.cols; x++){
                
                current_min = 255;

                for(int i = -(size/2); i <= (size/2); i++){
                    for(int j = -(size/2); j <= (size/2); j++){
                        current_min = min(current_min, padded.at<uchar>(y + (size/2) + i,x + (size/2) + j));
                    }
                }

                output_image.at<uchar>(y, x) = current_min;
            }
            
        }
        break;
    case MAX:
        uchar current_max;
        
        for(int y = 0; y < input_image.rows; y++){
            for(int x = 0; x < input_image.cols; x++){
                
                current_max = 0;

                for(int i = -(size/2); i <= (size/2); i++){
                    for(int j = -(size/2); j <= (size/2); j++){
                        current_max = max(current_max, padded.at<uchar>(y + (size/2) + i,x + (size/2) + j));
                    }
                }

                output_image.at<uchar>(y, x) = current_max;
            }
        }
        break;
    case MEDIAN:
        
        for(int y = 0; y < input_image.rows; y++){
            for(int x = 0; x < input_image.cols; x++){
                
                buffer.clear();

                for(int i = -(size/2); i <= (size/2); i++){
                    for(int j = -(size/2); j <= (size/2); j++){
                        buffer.push_back(padded.at<uchar>(y + (size/2) + i, x + (size/2) + j));
                    }
                }

                sort(buffer.begin(), buffer.end());
                
                output_image.at<uchar>(y, x) = buffer[buffer.size()/2];
            }
        }
        break;
    
    default:
        break;
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
    
    // args handling
    if (argc < 3 ) {
        std::cerr << "Usage: " << argv[0] << " <mode: G/E/S> <Filter dimension: D> <input_image_path> -p [params] [<output_image_path>]" << std::endl;
        return -1;
    }

    char mode = cv::toLowerCase(argv[1])[0];
    char *source_path = argv[3];
    char *output_path = 0;
    int filterSize = atoi(argv[2]);
    cv::Mat source_image = cv::imread(source_path, cv::IMREAD_COLOR);

    cv::Mat source_gray_image;
    cv::cvtColor(source_image, source_gray_image, cv::COLOR_BGR2GRAY);


    if(mode == 'g'){
        if (argc < 6 ) {
            std::cerr << "Usage: " << argv[0] << " <mode: G> <Filter dimension: D> <input_image_path> -p <sigma> [<output_image_path>]" << std::endl;
            return -1;
        }


        double sigma = atof(argv[5]);
        cv::Mat filtered = cv::Mat::zeros(source_gray_image.size(), source_gray_image.type());
        GaussianFilter(source_gray_image, filtered, filterSize, sigma);
        cv::imshow("Source", source_gray_image);
        cv::imshow("Blurred", filtered);
        cv::imshow("Difference", source_gray_image - filtered);
        if (argc > 6) output_path = argv[6];

        save_image(filtered, output_path);
    }

    else if(mode == 'e'){
        if (argc < 7 ) {
            std::cerr << "Usage: " << argv[0] << " <mode: E> <Filter dimension: D> <input_image_path> -p <x-direction flag> <y-direction flag> <threshold> [<output_image_path>]" << std::endl;
            return -1;
        }

        int threshold = atoi(argv[6]);
        cv::Mat edges = cv::Mat::zeros(source_gray_image.size(), source_gray_image.type());
        int xd = atoi(argv[4]);
        int yd = atoi(argv[5]);
        if (argc > 8) output_path = argv[7];
        edgeDetection(source_gray_image, edges, filterSize, threshold, yd,xd);
        cv::imshow("Source", source_gray_image);
        cv::imshow("Edges", edges);
        cv::imshow("Difference", source_gray_image - edges);
        save_image(edges, output_path);
    }

    else if(mode == 's'){
        if (argc < 6 ) {
            std::cerr << "Usage: " << argv[0] << " <mode: S> <Filter dimension: D> <input_image_path> -p <MIN/MAX/MEDIAN> [<output_image_path>]" << std::endl;
            return -1;
        }

        cv::Mat filtered = cv::Mat::zeros(source_gray_image.size(), source_gray_image.type());
        string filterMode = cv::toLowerCase(argv[5]);
        
        if (filterMode.compare("max") == 0){
            SelectionFilter(source_gray_image, filtered,  filterSize, MAX);
        }
        else if (filterMode.compare("min") == 0){
            SelectionFilter(source_gray_image, filtered,  filterSize, MIN);
        }
        else if (filterMode.compare("median") == 0){
            SelectionFilter(source_gray_image, filtered,  filterSize, MEDIAN);
        }
        else{   
            std::cerr << "Invalid mode." << std::endl;
            return -1;
        }
        cv::imshow("Source", source_gray_image);
        cv::imshow("Filtered", filtered);
        cv::imshow("Difference", source_gray_image - filtered);
        if (argc > 6) output_path = argv[6];

        save_image(filtered, output_path);

    }
    else{            
        std::cerr << "Invalid mode." << std::endl;
        return -1;
    }    

    cv::waitKey(0);
    cv::destroyAllWindows();

    exit(0);
}
