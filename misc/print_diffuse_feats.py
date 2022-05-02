import core.metrics as Metrics
import os

def print_feats(opt, train_data, feats_A, feats_B, level, t):
    feature_path = '{}/features'.format(opt['path']['results'])
    os.makedirs(feature_path, exist_ok=True)

    img_A = Metrics.tensor2img(train_data["A"][0,:,:,:])  # uint8
    Metrics.save_img(img_A, '{}/img_A.png'.format(feature_path))

    img_B = Metrics.tensor2img(train_data["B"][0,:,:,:])  # uint8
    Metrics.save_img(img_B, '{}/img_B.png'.format(feature_path))

    for i in range(feats_A[level].size(1)):
        feat_img_A = Metrics.tensor2img(feats_A[level][0,i,:,:])  # uint8
        Metrics.save_feat(feat_img_A, '{}/feat_A_{}_level_{}_t_{}.png'.format(feature_path, i, level, t))

        feat_img_B = Metrics.tensor2img(feats_B[level][0,i,:,:])  # uint8
        Metrics.save_feat(feat_img_B, '{}/feat_B_{}_level_{}_t_{}.png'.format(feature_path, i, level, t))
