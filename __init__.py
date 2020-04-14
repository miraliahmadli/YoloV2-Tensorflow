import sys
import numpy as np
import cv2 as cv2
import time
from yolov2tiny import YOLO_V2_TINY, postprocessing

def open_video_with_opencv(in_video_path, out_video_path):
    #
    # This function takes input and output video path and open them.
    #
    # Your code from here. You may clear the comments.
    #
    #print('open_video_with_opencv is not yet implemented')
    #sys.exit() 

    # Open an object of input video using cv2.VideoCapture.
    input_video = cv2.VideoCapture(in_video_path)    

    # if input_video.isOpened(): 
    #     # get input_video property 
    #     width  = input_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)  # float
    #     height = input_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float

    #     # or
    #     width  = input_video.get(3) # float
    #     height = input_video.get(4) # float

    #     # it gives me 0.0 :/
    #     fps = input_video.get(cv2.cv.CV_CAP_PROP_FPS)

    # Open an object of output video using cv2.VideoWriter.
    output_video = cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (416, 416))

    # Return the video objects and anything you want for further process.
    return input_video, output_video

def resize_input(im):
    imsz = cv2.resize(im, (416, 416))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return np.asarray(imsz, dtype=np.float32)

def video_object_detection(in_video_path, out_video_path, proc="cpu"):
    #
    # This function runs the inference for each frame and creates the output video.
    #
    # Your code from here. You may clear the comments.
    #
    # print('video_object_detection is not yet implemented')
    # sys.exit()

    # Open video using open_video_with_opencv.
    input_video, output_video = open_video_with_opencv(in_video_path, out_video_path)
    in_shape = (1, 416, 416, 3)
    pickle_path = "./y2t_weights.pickle"

    # Check if video is opened. Otherwise, exit.
    if not input_video.isOpened():
        print('video is not opened')
        sys.exit()
    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.
    model = YOLO_V2_TINY(in_shape, pickle_path, proc)

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.
    while True:
        ret, img = input_video.read()
        if not ret:
            continue
        img = resize_input(img)
        print(img.shape)

        start = time.time()
        output_tensors = model.inference(img)
        end = time.time()
        elapsed_time = end-start
        print("Elapsed time to run inference: {}".format(elapsed_time))

        print("\n\nLength and type", len(output_tensors), type(output_tensors))
        break

        label_boxes = postprocessing(output_tensors[-1])
    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.
    

    # Release the opened videos.
    

def main():
    if len(sys.argv) < 3:
        print ("Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])")
        sys.exit()

    in_video_path = sys.argv[1] 
    out_video_path = sys.argv[2] 

    if len(sys.argv) == 4:
        proc = sys.argv[3]
    else:
        proc = "cpu"

    video_object_detection(in_video_path, out_video_path, proc)

if __name__ == "__main__":
    main()
