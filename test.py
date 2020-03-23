import torch
from PIL import Image
import parser
import models
import models_best
import data_c
import numpy as np
import mean_iou_evaluate
import os
import data_test


def evaluate(model_in, data_loader):
    # set model to evaluate mode
    model_in.eval()
    pred_list = []
    gts = []

    print("Training done, Entered evaluate function")
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (images, gt) in enumerate(data_loader):
            images = images.cuda()
            pred = model_in(images)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()

            gt = gt.numpy().squeeze()

            pred_list.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    pred_list = np.concatenate(pred_list)

    np.save("pred_list.npy", pred_list)

    return mean_iou_evaluate.mean_iou_score(pred_list, gts)


if __name__ == '__main__':
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data_test.DataTest(args, mode='test'),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' prepare mode '''
    print('===> load model ...')
    if(args.resume == 'model_best.pth-12.tar?dl=1'):
        model = models.Net(args).cuda()
    else:
        model = models_best.Net(args).cuda()

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    model.eval()

    pred_list = []

    # acc = evaluate(model, test_loader)
    # print('Testing Accuracy: {}'.format(acc))
    print('===> make predictions ...')
    with torch.no_grad():
        for idx, images in enumerate(test_loader):
            images = images.cuda()
            pred = model(images)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            pred_list.append(pred)

        pred_list = np.concatenate(pred_list)

        for idx, pred_img in enumerate(pred_list):
            if idx < 10:
                imgs_directory = os.path.join(args.save_dir, '000' + str(idx) + '.png')
            elif idx < 100:
                imgs_directory = os.path.join(args.save_dir, '00' + str(idx) + '.png')
            else:
                imgs_directory = os.path.join(args.save_dir, '0' + str(idx) + '.png')

            img = Image.fromarray(pred_img.astype('uint8'))
            img.save(imgs_directory)