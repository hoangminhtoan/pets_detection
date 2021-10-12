#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <fstream>
#include <string>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>

using namespace std;
using namespace cv;

const string yolov3_config = {
    "   name: \"yolov4_voc\" \n"
    "   model_type : YOLOv3 \n"
    "   yolo_v3_param { \n"
    "     num_classes: 2 \n"
    "     anchorCnt: 3 \n"
    "     conf_threshold: 0.25 \n"
    "     nms_threshold: 0.45 \n"
    "     layer_name: \"29\" \n"
    "     layer_name: \"36\" \n"
    "     layer_name: \"43\" \n"
    "     biases: 344 \n"
    "     biases: 319 \n"
    "     biases: 135 \n"
    "     biases: 169 \n"
    "     biases: 81 \n"
    "     biases: 82 \n"
    "     biases: 37  \n"
    "     biases: 58  \n"
    "     biases: 23  \n"
    "     biases: 27  \n"
    "     biases: 10  \n"
    "     biases: 14 \n"
    "     test_mAP: false \n"
    "   } \n"};
struct ID_XYXY {
    int cls;
    float confidence_score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};
std::string read_config(string model_folder) {
    std::ifstream cfg_file(model_folder + "./prototxt");
    std::string cfg;
    std:string line;
    while (cfg_file >> line) {
        cfg += line;
    }
    return cfg;
}
std::vector<std::string> read_benchmark_video(std::string benchmark_list) {
    std::ifstream benchmark_list_file(benchmark_list);
    std::vector<std::string> list = {};
    std::string l;
    while (benchmark_list_file >> l) {
        list.push_back(l);
    }
    return list;
}

// infer_batch but batch size is currently 1
void infer_batch_1(string video_out_path, int image_id, std::unique_ptr<vitis::ai::DpuTask>& task,
        std::vector<vitis::ai::library::InputTensor>& input_tensor,
        vitis::ai::proto::DpuModelParam& config,
        std::vector<std::vector<ID_XYXY>>& imgs_boxes,
        std::vector<cv::Mat>& org_inputs,
        std::vector<cv::Mat>& inputs,
        std::vector<int>& input_cols,
        std::vector<int>& input_rows) {
    
    
    // Set the input images into dpu.
    task->setImageRGB(inputs);
    /* DPU Runtime */
    // Run the dpu.
    task->run(0u);
    /* Post-process part */
    // Get output.
    auto output_tensor = task->getOutputTensor(0u);
    // Execute the yolov3 post-processing.
    // cout << "111111" << endl;
    // int sWidth = input_tensor[0].width;
    // int sHeight = input_tensor[0].height;
    // auto num_classes = config.yolo_v3_param().num_classes();
    // auto nms_thresh = config.yolo_v3_param().nms_threshold();
    // auto conf_thresh = config.yolo_v3_param().conf_threshold();
    // auto mAP = config.yolo_v3_param().test_map();
    // cout << sWidth << " " << sHeight << " " << num_classes << " " << nms_thresh << " " << conf_thresh << " " << mAP << endl;
    auto results = vitis::ai::yolov3_post_process(
        input_tensor, output_tensor, config, input_cols, input_rows);
    // cout << "222222" << endl;
    // Convert coordinate and draw boxes at origin image.
    // also save image to debugging purpose
    for (int k = 0; k < (int)inputs.size(); k++) {
        std::vector<ID_XYXY> boxes = {};
        for (auto& box : results[k].bboxes) {
            int label = box.label;
            float xmin = box.x * input_cols[k] + 1;
            float ymin = box.y * input_rows[k] + 1;
            float xmax = xmin + box.width * input_cols[k];
            float ymax = ymin + box.height * input_rows[k];
            if (xmin < 0.) xmin = 1.;
            if (ymin < 0.) ymin = 1.;
            if (xmax > input_cols[k]) xmax = input_cols[k];
            if (ymax > input_rows[k]) ymax = input_rows[k];
            float confidence = box.score;
            boxes.push_back(ID_XYXY{label, confidence, xmin, ymin, xmax, ymax});
            cv::rectangle(org_inputs[k], cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                          cv::Scalar(0, 255, 0), 2, 1, 0);
        }
        imgs_boxes.push_back(boxes);
        if ((int)imgs_boxes[0].size() > 0) {
            cv::imwrite(video_out_path + "/" + to_string(image_id) + ".jpeg", org_inputs[k]);
        }
    }
}
int infer_batch(int step_size, string video_out_path, std::unique_ptr<vitis::ai::DpuTask>& task,
        std::vector<vitis::ai::library::InputTensor>& input_tensor,
        vitis::ai::proto::DpuModelParam& config,
        std::vector<cv::Mat>& org_inputs,
        std::vector<cv::Mat>& inputs,
        std::vector<int>& input_cols,
        std::vector<int>& input_rows,
        std::vector<int>& input_ids) {
    string cmd = "mkdir -p " + video_out_path;
    std::system(cmd.c_str());
    int cnt = 0;
    for (int i = 0; i < (int)inputs.size(); i += step_size) {
        std::vector<cv::Mat> t_org_inputs = {org_inputs[i]};
        std::vector<cv::Mat> t_inputs = {inputs[i]};
        std::vector<int> t_input_cols = {input_cols[i]};
        std::vector<int> t_input_rows = {input_rows[i]};
        std::vector<std::vector<ID_XYXY>> imgs_boxes = {};
        infer_batch_1(video_out_path, input_ids[i], task, input_tensor, config,
            imgs_boxes,
            t_org_inputs,
            t_inputs,
            t_input_cols,
            t_input_rows);
        if ((int)imgs_boxes[0].size() > 0) {
            cnt++;
        }
        t_inputs.clear();
        t_input_cols.clear();
        t_input_rows.clear();
        imgs_boxes.clear();
    }
    return cnt;
}
int main(int argc, char* argv[]) {
    // init model on DPU
    // A kernel name, it should be samed as the dnnc result. e.g.
    // /usr/share/vitis_ai_library/models/yolov3_voc/yolov3_voc.elf
    string model_name = (string)argv[1] + "/" + (string)argv[1];
    string kernel_name = model_name + ".xmodel";
    cout << kernel_name << endl;
    // Create a dpu task object.
    std::unique_ptr<vitis::ai::DpuTask> task = vitis::ai::DpuTask::create(kernel_name);
    auto batch = task->get_input_batch(0, 0);
    // Set the mean values and scale values.
    task->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                            {0.00390625f, 0.00390625f, 0.00390625f});
    cout << "batch " << batch << endl;
    std::vector<vitis::ai::library::InputTensor> input_tensor = task->getInputTensor(0u);
    CHECK_EQ((int)input_tensor.size(), 1)
        << " the dpu model must have only one input";
    auto width = input_tensor[0].width;
    auto height = input_tensor[0].height;
    auto size = cv::Size(width, height);
    int block_size = 10;
    int step_size = 1;
    cout << "Width " << width << "  " << "Height " << height << endl;
    // Create a config and set the correlating data to control post-process.
    std::string config_file = model_name + ".prototxt";
    vitis::ai::proto::DpuModelParam config;
    // Fill all the parameters.
    auto ok =
        google::protobuf::TextFormat::ParseFromString(yolov3_config, &config);
    if (!ok) {
        cerr << "Set parameters failed!" << endl;
        abort();
    }

    // Create a VideoCapture object and open the input file
    std::string benchmark_list_file = argv[2];
    std::vector<std::string> benchmark_list = read_benchmark_video(benchmark_list_file);
    for (std::string f: benchmark_list) {
        if (f.find("mp4") == -1) continue;
        string video_out_path = "/home/root/pet/saved_frames/" + f;
        string cmd = "rm -rf " + video_out_path;
        std::system(cmd.c_str());
        bool detected = false;
        // cout << f << endl;
        cv::VideoCapture cap(f);
        // Check if camera opened successfully
        if(!cap.isOpened()){
            cout << "Error opening video stream or file" << endl;
            return -1;
        }
        
        int image_id = 0;
        std::vector<cv::Mat> org_inputs;
        std::vector<cv::Mat> inputs;
        std::vector<int> input_cols, input_rows, input_ids;
        while (true){
            cv::Mat frame;
            cv::Mat image;
            // Capture frame-by-frame
            cap >> frame;
        
            // If the frame is empty, break immediately
            if (frame.empty())
                break;
            
            org_inputs.push_back(frame);
            input_ids.push_back(image_id);
            input_cols.push_back(frame.cols);
            input_rows.push_back(frame.rows);
            if (size != frame.size()) {
                cv::resize(frame, image, size);
            } else {
                image = frame;
            }
            inputs.push_back(image);
            // if ((int)inputs.size() == batch) {
            //     // infer
            //     std::vector<std::vector<ID_XYXY>> imgs_boxes;
            //     infer_batch_1(task,
            //                 input_tensor, config, imgs_boxes,
            //                 inputs, input_cols, input_rows);
            //     if ((int)imgs_boxes.size() > 0) {
            //         detected = true;
            //     }
            //     imgs_boxes.clear();
            //     inputs.clear();
            //     input_cols.clear();
            //     input_rows.clear();
            // }
            if ((int)inputs.size() == block_size) {
                int cnt = infer_batch(step_size,
                            video_out_path,
                            task, input_tensor, config,
                            org_inputs, inputs, input_cols, input_rows, input_ids);
                if (cnt > 2) {
                    detected = true;
                }
                org_inputs.clear();
                inputs.clear();
                input_ids.clear();
                input_cols.clear();
                input_rows.clear();
            }
            image_id++;
        }
        // When everything done, release the video capture object
        cap.release();
        if (detected) {
            cout << f << " True" << endl;
        } else {
            cout << f << " False" << endl;
        }
    }
    return 0;
}