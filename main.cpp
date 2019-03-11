#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/opencv/cv_image.h>


int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "can not open camera" << std::endl;
        return EXIT_FAILURE;

    }

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shapePredictor;
    dlib::deserialize(argv[1]) >> shapePredictor;

    dlib::image_window window;

    while (!window.is_closed())
    {
        cv::Mat cvFrame;
        dlib::array2d<dlib::rgb_pixel> dlibFrame;

        if (cap.read(cvFrame))
        {
             dlib::assign_image(dlibFrame, dlib::cv_image<dlib::bgr_pixel>(cvFrame));
             std::vector<dlib::rectangle> rectangles = detector(dlibFrame);
             std::vector<dlib::full_object_detection> shapes;

             for (unsigned long i = 0; i < rectangles.size(); ++i)
             {
               shapes.push_back(shapePredictor(dlibFrame, rectangles[i]));
             }
             window.clear_overlay();
             window.set_image(dlibFrame);
             //window.add_overlay(shapes);
             window.add_overlay(dlib::render_face_detections(shapes));
        }

    }
    return  EXIT_SUCCESS;
}
