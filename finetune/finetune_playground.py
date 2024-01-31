import torch 

import sys
sys.path.append('.')
from src.models.vision_transformer import vit_huge
from src.models.mae_vit import vit_huge_patch14


if __name__=="__main__":
    #p = 'finetune/model_output_final.txt'
    #model = vit_huge_patch14()
    #print(model.modules)
    """
    for name, param in model.named_parameters():
        with open(p, mode='a') as f:
            f.write(f"Parameter name: {name}, Shape: {param.shape}\n")
    """
    checkpoint_path = "pre-training_weights/IN1K-vit.h.14-300e.pth.tar"
    model = vit_huge(patch_size=14, drop_path_rate=0.2)

    state_dict = model.state_dict()

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu'
    )

    checkpoint_model = checkpoint['target_encoder']

    new_checkpoint = {key.replace('module.', ''): value for key, value in checkpoint_model.items()}

    """
    for key, value in new_checkpoint.items():
        print(f"Key: {key}, Shape: {value.shape}")
    """
    
    model.load_state_dict(new_checkpoint, strict=False)
    """
    model_without_ddp = model.modules
    print(len(model_without_ddp.blocks))
    """

    model_without_ddp = model.modules
    print(model.named_parameters())
    
    """
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    """