#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

static vector<string> loadNames(const string& path){
    vector<string> names;
    ifstream f(path);
    string s;
    while (getline(f,s)) names.push_back(s);
    return names;
}

static void drawDetections(Mat& frame, const vector<int>& classIds,
                           const vector<float>& confs, const vector<Rect>& boxes,
                           const vector<string>& names){
    for(size_t i=0;i<boxes.size();++i){
        rectangle(frame, boxes[i], Scalar(0,255,0), 2);
        string label = (classIds[i] < (int)names.size()? names[classIds[i]]: to_string(classIds[i]));
        label += format(" %.2f", confs[i]);
        int baseLine=0;
        Size t = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int y = max(0, boxes[i].y - t.height - baseLine - 6);
        rectangle(frame, Rect(boxes[i].x, y, t.width+6, t.height+baseLine+6), Scalar(0,255,0), FILLED);
        putText(frame, label, Point(boxes[i].x+3, y+t.height+2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1);
    }
}

int main(){
    string cfg="yolov4-tiny.cfg";
    string weights="yolov4-tiny.weights";
    string namesPath="coco.names";

    auto names = loadNames(namesPath);

    dnn::Net net = dnn::readNetFromDarknet(cfg, weights);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    VideoCapture cap(0);
    if(!cap.isOpened()){
        cerr << "ERROR: cannot open camera (index 0)\n";
        return -1;
    }

    const int inpSize = 320;
    const float confTh = 0.35f;
    const float nmsTh  = 0.45f;

    double avgInf=0; int cnt=0;

    while(true){
        Mat frame;
        cap >> frame;
        if(frame.empty()) break;

        auto t0 = chrono::high_resolution_clock::now();

        auto p0 = chrono::high_resolution_clock::now();
        Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(inpSize, inpSize),
                                      Scalar(), true, false);
        net.setInput(blob);
        auto p1 = chrono::high_resolution_clock::now();

        vector<Mat> outs;
        auto i0 = chrono::high_resolution_clock::now();
        net.forward(outs, net.getUnconnectedOutLayersNames());
        auto i1 = chrono::high_resolution_clock::now();

        auto o0 = chrono::high_resolution_clock::now();
        vector<int> classIds;
        vector<float> confs;
        vector<Rect> boxes;

        for(auto &out : outs){
            for(int r=0; r<out.rows; r++){
                float* data = (float*)out.ptr(r);
                float obj = data[4];
                if(obj < confTh) continue;

                Mat scores = out.row(r).colRange(5, out.cols);
                Point classIdPoint;
                double maxClass;
                minMaxLoc(scores, 0, &maxClass, 0, &classIdPoint);

                float conf = obj * (float)maxClass;
                if(conf < confTh) continue;

                int cx = (int)(data[0] * frame.cols);
                int cy = (int)(data[1] * frame.rows);
                int w  = (int)(data[2] * frame.cols);
                int h  = (int)(data[3] * frame.rows);
                int x  = cx - w/2;
                int y  = cy - h/2;

                classIds.push_back(classIdPoint.x);
                confs.push_back(conf);
                boxes.push_back(Rect(x,y,w,h));
            }
        }

        vector<int> idx;
        dnn::NMSBoxes(boxes, confs, confTh, nmsTh, idx);

        vector<int> fClass; vector<float> fConf; vector<Rect> fBox;
        for(int id : idx){
            fClass.push_back(classIds[id]);
            fConf.push_back(confs[id]);
            fBox.push_back(boxes[id]);
        }
        auto o1 = chrono::high_resolution_clock::now();

        drawDetections(frame, fClass, fConf, fBox, names);

        auto t1 = chrono::high_resolution_clock::now();

        double pre_ms  = chrono::duration<double, milli>(p1-p0).count();
        double inf_ms  = chrono::duration<double, milli>(i1-i0).count();
        double post_ms = chrono::duration<double, milli>(o1-o0).count();
        double tot_ms  = chrono::duration<double, milli>(t1-t0).count();

        avgInf += inf_ms; cnt++;

        cout << "pre=" << pre_ms << "ms "
             << "inf=" << inf_ms << "ms "
             << "post="<< post_ms<< "ms "
             << "tot=" << tot_ms << "ms "
             << "avg_inf=" << (avgInf/cnt) << "ms "
             << "FPS=" << (1000.0/tot_ms) << "\n";

        imshow("YOLOv4-tiny CPU 320 (Windows/MSYS2)", frame);
        if(waitKey(1) == 27) break; // ESC
    }
    return 0;
}
