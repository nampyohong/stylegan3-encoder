import torch
from training.networks_irse import Backbone


l2_criterion = torch.nn.MSELoss(reduction='mean')


def l2_loss(generated_images, real_images):
    return l2_criterion(generated_images, real_images)


class IDLoss(torch.nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        self.facenet = Backbone(
            input_size=112,
            num_layers=50,
            drop_ratio=0.6,
            mode='ir_se'
        )
        # TODO : get pretrained weight path from config
        self.facenet.load_state_dict(torch.load('pretrained/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112,112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, generated_images, y, x): # y == x for image inversion
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        generated_feats = self.extract_feats(generated_images)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = generated_feats[i].dot(y_feats[i])
            diff_input = generated_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count
