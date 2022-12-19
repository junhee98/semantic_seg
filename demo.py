import os 
import cv2
import argparse
import numpy as np
import shutil
from tqdm import tqdm

def main(input_path, output_path):

    sub_infer = 'infer'
    sub_output = 'outputs'

    input_list = sorted(os.listdir(input_path))
    output_list = sorted(os.listdir(os.path.join(output_path,sub_infer)))

    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)    

    if not os.path.exists(os.path.join(output_path,sub_output)):
        os.makedirs(os.path.join(output_path,sub_output))


    #Phase1 : input image & infer image concat / visualization 
    print("Phase1")
    for input, infer in tqdm(zip(input_list,output_list)):
        input_image = cv2.imread(os.path.join(input_path,input),cv2.IMREAD_COLOR)
        infer_image = cv2.imread(os.path.join(output_path,sub_infer,infer),cv2.IMREAD_COLOR)

        if input_image.shape != infer_image.shape:
            input_image = cv2.resize(input_image,(infer_image.shape[1],infer_image.shape[0]),interpolation=cv2.INTER_LINEAR)

        output_image = [input_image,infer_image]
        #print(input_image.shape)
        #print(infer_image.shape)
        cv2.imshow("Image",np.hstack(output_image))
        cv2.imwrite(os.path.join(output_path,sub_output,infer),np.hstack(output_image))

        if cv2.waitKey(1) == 27:
            break
    
    h, w, _ = input_image.shape

    cv2.destroyAllWindows()


    #Phase2 : concat image to Video
    print("Phase2")
    frame_size = (int(w*2),h)

    output_list = sorted(os.listdir(os.path.join(output_path,sub_output)))

    out = cv2.VideoWriter(os.path.join(output_path,'result.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 30, frame_size)

    for list in tqdm(output_list):
        output = cv2.imread(os.path.join(output_path,sub_output,list),cv2.IMREAD_COLOR)
        out.write(output)

    out.release()

    #Phase3 : remove dummy directory
    print("Phase3")
    shutil.rmtree(os.path.join(os.path.join(output_path,sub_output)))
    print("Done!")


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('-ip', '--input_path', default=None,type=str,
                        help='Path to the Image directory')
    parser.add_argument('-op', '--output_path', default=None,type=str,
                        help='Path to the Image directory')
    args = parser.parse_args()
    
    main(args.input_path,args.output_path)