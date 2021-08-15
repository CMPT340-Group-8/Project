from argparse import ArgumentParser
import mmcv
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
from tqdm import tqdm
from ipdb import set_trace
def main():
    parser = ArgumentParser()
    parser.add_argument( '--img_path', default='./pred/', help='Image file path')
    parser.add_argument('--config', default='./model/faster_rcnn_r50_fpn_1x_coco_tumor.py', help='Config file')
    parser.add_argument('--checkpoint', default='./model/epoch_18.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.9, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for img in tqdm(os.listdir(args.img_path)):

        # test a single image
        result = inference_detector(model, args.img_path + img)
        # show the results
        result_pic = show_result_pyplot(model, args.img_path + img, result, score_thr=args.score_thr)
        mmcv.imwrite(result_pic, 'result/' + img)



if __name__ == '__main__':
    main()
