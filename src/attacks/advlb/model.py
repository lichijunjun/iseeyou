import torch
from torchvision.models import resnet50


def load_advlb_model(model_type, device="cuda"):
    if model_type == 'resnet50':
        print("Loading model...")
        model = resnet50(pretrained=True)
    elif model_type == 'df_resnet50':
        print("Loading adv trained model...")
        model = resnet50(pretrained=False)
        model.load_state_dict(torch.load('./model/checkpoint-89.pth.tar')['state_dict'])
    else:
        raise ValueError(f"Only `resnet50` and `df_resnet50` supported for AdvLB attack")
    model.to(device)
    model.eval()
    return model