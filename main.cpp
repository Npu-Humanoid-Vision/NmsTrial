#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

// 摄像机模式/图片模式
#define CAMERA_MODE
// #define PIC_MODE

// 是否在达尔文上跑
// 可用 CV_MAJOR_VERSION代替
// #define RUN_ON_DARWIN

// 在摄像机模式获得样本
#define GET_SAMPLE_LIVING

// #define MODEL_NAME "../SvmTrain/ball_linear_auto.xml"
#define MODEL_NAME "../SvmTrain/nu_svr_linear.xml"

#define IMG_COLS 32
#define IMG_ROWS 32

// 开启选项之后
#ifdef GET_SAMPLE_LIVING
    #define POS_COUNTER_INIT_NUM 384
    #define NEG_COUNTER_INIT_NUM 160
    #define SAVE_PATH "../../BackUpSource/Ball/Train/"

    int pos_counter = POS_COUNTER_INIT_NUM;
    int neg_counter = NEG_COUNTER_INIT_NUM;

    string GetPath(string save_path, int lable) {
        stringstream t_ss;
        string t_s;

        if (lable == POS_LABLE) {
            save_path += "Pos/";
            t_ss << pos_counter++;
            t_ss >> t_s;
            t_s = save_path + t_s;
            t_s += ".jpg";
            cout<<t_s<<endl;
        }
        else {
            save_path += "Neg/";
            t_ss << neg_counter++;
            t_ss >> t_s;
            t_s = save_path + t_s;
            t_s += ".jpg";
            cout<<t_s<<endl;   
        }
        return t_s;
    }
#endif

// NMS threshold
#define NMS_THRE 0.5

// define rect with scores
struct MyRect {
    MyRect() {
        valid = true;
    }
    cv::Rect rect;
    float scores;
    bool valid;
};
// for MyRect sort with scores
bool RectSort(MyRect r_1, MyRect r_2) {
    return r_1.scores > r_2.scores;
}

// get iou
double GetIou(MyRect r_1, MyRect r_2) {
    cv::Rect inter = r_1.rect | r_2.rect;
    return 1.0*inter.area()/(r_1.rect.area()+r_2.rect.area()-inter.area());
}

// perform nms
void Nms(std::vector<MyRect>& input_rects, double nms_thre) {
    sort(input_rects.begin(), input_rects.end(), RectSort);

    for (auto i = input_rects.begin(); i != input_rects.end(); i++) {
        for (auto j = input_rects.begin(); j != input_rects.end(); j++) {
            if (GetIou(*i, *j) > nms_thre) {
                j->valid = false;
            }   
        }
    }
}


int main(int argc, char const *argv[]) {
    // load SVM model
#if CV_MAJOR_VERSION < 3
    CvSVM tester;
    tester.load(MODEL_NAME);
#else
    cv::Ptr<cv::ml::SVM> tester = cv::ml::SVM::load(MODEL_NAME);
#endif
#ifdef CAMERA_MODE
    cv::VideoCapture cp(0);
    cv::Mat frame; 
    cv::Rect ROI_Rect(100, 100, 9*IMG_COLS, 9*IMG_ROWS);

    cp >> frame;
    while (frame.empty()) {
        cp >> frame;
    }
    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr << __LINE__ <<"frame empty"<<endl;
            return -1;
        }
#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        cv::Mat ROI = frame(ROI_Rect).clone();
        cv::resize(ROI, ROI, cv::Size(IMG_COLS, IMG_ROWS));
        cv::HOGDescriptor hog_des(Size(IMG_COLS, IMG_ROWS), Size(16,16), Size(2,2), Size(8,8), 12);
        std::vector<float> hog_vec;
        hog_des.compute(ROI, hog_vec);

        cv::Mat t(hog_vec);
        cv::Mat hog_vec_in_mat = t.t();
        hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);
#if CV_MAJOR_VERSION < 3
        int lable = (int)tester.predict(hog_vec_in_mat);
        cout<<tester.predict(hog_vec_in_mat)<<endl;
        if (lable == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
#else
        cv::Mat lable;
        tester->predict(hog_vec_in_mat, lable);
        if (lable.at<float>(0, 0) == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        } 
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
#endif
        
        cv::imshow("frame", frame);
        char key = cv::waitKey(20);
        if (key == 'q') {
            break;
        }
        else if (key == 'p') {
            cv::imwrite(GetPath(SAVE_PATH, POS_LABLE), ROI);
        }
        else if (key == 'n') {
            cv::imwrite(GetPath(SAVE_PATH, NEG_LABLE), ROI);
        }

    }

#endif

#ifdef PIC_MODE
    // load Test Data
    std::vector<cv::Mat> test_image;
    cv::Mat test_data;
    int test_images_num;
    
    HogGetter hog_getter;
    hog_getter.ImageReader_("D:/baseRelate/code/svm_trial/BackUpSource/People/Test/HSample/", "/*.jpg");
    test_data = hog_getter.HogComputter_();
    test_image = hog_getter.raw_images_;
    test_images_num = hog_getter.sample_nums_;
    // GetImage("D:/baseRelate/code/svm_trial/BackUpSource/People/Test/PosSample/", "*.jpg", test_image, test_data);
    // for (auto i = test_image.begin(); i != test_image.end(); i++) {
    //     cv::imshow("233", *i);
    //     cv::waitKey();
    // }
    
    // Compute correct rate
    // cout<<test_images_num<<endl;
    int correct_nums = 0;
    for (int i=0; i<test_images_num; i++) {
        // t = test_data.rowRange(i, i+1);
        cv::Mat t = test_data.row(i).clone();
        // cout<<t.size()<<endl;
        t.convertTo(t, CV_32FC1);
        int predict_lable = (int)tester.predict(t);
        // cout<<predict_lable<<endl;
        correct_nums += predict_lable;
    }
    cout<< correct_nums*1.0/test_images_num<<endl;
#endif
    return 0;
}
