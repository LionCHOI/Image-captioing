import torch
import torch.nn as nn
from torchvision.models import densenet161, resnet152, vgg19, vit_h_14

class Encoder(nn.Module):
    # init --------------------------------------------------------------------------
    def __init__(self, network='vit'):
        super(Encoder, self).__init__()

        # encoder choice ------------------------------------------------------------
        self.network = network      # 입력으로 받은 network 저장
        
        if network == 'resnet152':                                              # network가 resnet152이면
            self.net = resnet152(weights="ResNet152_Weights.DEFAULT")           # pretrained model을 가져와서
            self.net = nn.Sequential(*list(self.net.children())[:-2])           # FC layer 삭제
            self.dim = 2048                                                     # decoder에 넣을 입출력 dimension 저장
            
        elif network == 'densenet161':                                          # network가 densenet161이면
            self.net = densenet161(pretrained=True)                             # pretrained model을 가져와서
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])  # FC layer 삭제
            self.dim = 1920                                                     # decoder에 넣을 입출력 dimension 저장
            
        elif network == 'vit':                                                  # network가 vit이면
            self.model = vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')          # pretrained model을 가져와서
            self.encoder = nn.Sequential(*list(self.model.children())[:-1])[1]  # FC layer 삭제
            self.dim = 1280                                                     # decoder에 넣을 입출력 dimension 저장
            
        else:                                                                   # network가 vgg19이면
            self.net = vgg19(weights='VGG19_Weights.DEFAULT')                   # pretrained model을 가져와서
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])  # FC layer 삭제
            self.dim = 512                                                      # decoder에 넣을 입출력 dimension 저장

    # forward -------------------------------------------------------------------------
    def forward(self, x):
        # vit forward -----------------------------------------------------------------
        if self.network == 'vit':
            x = self.model._process_input(x)    # x를 patch로 나누는 작업 수행         
            n = x.shape[0]                      # batch size를 확보
            
            batch_class_token = self.model.class_token.expand(n, -1, -1)    # class token 생성
            x = torch.cat([batch_class_token, x], dim=1)                    # class token 연결
            
            x = self.encoder(x)     # encoder 수행
            x = x[:, :256, :]       # decoder에는 class token을 제외하고 넘긴다.
            
        # the others forward -----------------------------------------------------------
        else:
            x = self.net(x)                         # encoder를 수행
            x = x.permute(0, 2, 3, 1)               # decoder 입력에 알맞은 형태로 맞추기 위한 사전 작업 (위치 변경)
            x = x.view(x.size(0), -1, x.size(-1))   # decoder 입력에 알맞은 형태로 맞추기
            
        return x
