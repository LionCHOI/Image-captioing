import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
import argparse, json
from torch.autograd import Variable
from tqdm import tqdm

# model
from include.encoder import Encoder
from include.decoder import Seq2SeqDecoder, translate
from include.dataset import ImageCaptionDataset
from include.utils import *

# metric
import evaluate
metric_meteor = evaluate.load("meteor")
from nltk.translate.bleu_score import corpus_bleu

# global variable --------------------------------------------------------------------------------------------------
data_transforms = transforms.Compose([                  # Dataloader에 사용할 transform 설정 
    transforms.Resize((224, 224)),                      # transform 설정 - size를 (224, 224)로 설정
    transforms.ToTensor(),                              # transform 설정 - tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # transform 설정 - Normalize - 평균 값 설정
                         std=[0.229, 0.224, 0.225])     # transform 설정 - Normalize - 표준편차 값 설정
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda 설정

BOS_IDX=0   # start symbol index
EOS_IDX=1   # end symbol index
PAD_IDX=3   # padding index
MAX_LEN=50  # max length for prediction

# EVALUATION =============================================================================================================
def eval(args):
    # Data Loader   -------------------------------------------------------------------------
    word_dict = json.load(open(args.data + '/word_dict.json', 'r')) # vocab을 불러오는 부분
    vocabulary_size = len(word_dict)                                # vocab의 크기

    convert_word_dict = {v:k for k,v in word_dict.items()}          # embedding 값을 문자로 변환하기 위한 dictionary

    # model initialization  ----------------------------------------------------------------- 
    encoder = Encoder(args.network)                             # encoder 설정 (network의 값에 따라서 encoder를 변경 가능) ['vgg19', 'resnet152', 'densenet161', 'vit']
    decoder = Seq2SeqDecoder(d_model=encoder.dim,               # docoder의 입출력 차원 (encoder의 입출력 차원과 동일!)
                             nhead=8,                           # head의 개수
                             tgt_vocab_size=vocabulary_size,    # vocab의 크기
                             num_decoder_layers=12,             # decoder의 수
                             dim_feedforward=2048,              # feedforward의 수
                             dropout=0.1,                       # drop out rate
                             batch_first=True,                  # 첫 번째가 batch인가
                             device=device)                     # cuda 설정
    
    decoder.load_state_dict(torch.load(args.checkpoint))        # 미리 학습된 모델이 있으면 불러오기

    encoder.to(device)                                          # encoder model을 GPU에 넣기
    decoder.to(device)                                          # decoder model을 GPU에 넣기
    
    encoder.eval()                                              # encoder 부분을 eval()로 하여 학습하지 못하도록 한다. 
    decoder.eval()                                              # decoder 부분을 eval()로 하여 학습하지 못하도록 한다.

    # data setting  -------------------------------------------------------------------------
    val_loader = torch.utils.data.DataLoader(                   # valid loader 설정
        ImageCaptionDataset(data_transforms,                    # Image Caption Dataset 설정 - transform 설정
                            args.data,                          # Image Caption Dataset 설정 - data path를 넘겨줌
                            split_type='val'),                  # Image Caption Dataset 설정 - valid data로 설정
        batch_size=args.batch_size,                             # batch size 설정 (무조건 1로 고정 -- prediction마다 <EOS>가 달라서)
        shuffle=False,                                          # Shuffle 설정    
        num_workers=1)                                          # worker 수 설정

    # BLEU score를 위한 변수 설정   -----------------------------------------------------------------
    references_word, references_token = [], []   # real value를 위한 list (word, token)
    hypotheses_word, hypotheses_token = [], []   # pred value를 위한 list (word, token)

    # processing part ------------------------------------------------------------------------------
    print('Starting Evaluation with {}'.format(args)) # train 시작 부분 터미널에 출력
    with torch.no_grad():                                                                                           # gradient 계산하지 않도록 설정
        for batch_idx, (imgs, captions, all_captions) in enumerate(tqdm(val_loader, total=len(val_loader)), 1):     # for문 수행 with tqdm
            if batch_idx % 5 == 0:  # 5번씩 건너뛰기 (현재 저는 한 이미지당 5개의 caption을 받기에 이를 조절하고자 5개씩 건너뜁니다.)                                                                                  
                # data load & preprocessing --------------------------------------------------------
                imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)   # data를 대입한다.
                
                # processing    ----------------------------------------------------------------------------
                img_features = encoder(imgs)                    # image를 encoder에 넣어 image feature 획득
                
                preds = translate(model=decoder,                # 번역 수행 - 변역 수행에 사용할 decoder model
                                  img_feature=img_features,     # 번역 수행 - 변역 수행에 사용할 image feature
                                  max_len=MAX_LEN,              # 번역 수행 - 변역 수행에 사용할 최대 길이
                                  start_symbol=BOS_IDX,         # 번역 수행 - 변역 수행에 사용할 시작 token 인덱스
                                  end_symbol=EOS_IDX)           # 번역 수행 - 변역 수행에 사용할 끝 token 인덱스
                
                # preprocessing for evaluation ------------------------------------------------------------
                for idxs in preds.tolist():
                    hypotheses_token.append([idx for idx in idxs            # BLEU에 사용할 token으로 구성된 prediction 수집
                                        if idx != word_dict['<start>']      # 만약 <start>면 제외
                                        and idx != word_dict['<pad>']       # 만약 <pad>면 제외
                                        and idx != word_dict['<eos>']])     # 만약 <eos>면 제외
                    
                    hypotheses_word.append(' '.join([convert_word_dict[idx] for idx in idxs     # METEOR에 사용할 word로 구성된 prediction 수집
                                                     if idx != word_dict['<start>']             # 만약 <start>면 제외
                                                     and idx != word_dict['<pad>']              # 만약 <pad>면 제외
                                                     and idx != word_dict['<eos>']]))           # 만약 <eos>면 제외
                    
                for cap_set in all_captions.tolist():
                    caps_word, caps_token = [], []
                    for caption in cap_set:
                        
                        caps_token.append([word_idx for word_idx in caption         # BLEU에 사용할 token으로 구성된 Ground Truth value 수집
                                           if word_idx != word_dict['<start>']      # 만약 <start>면 제외
                                           and word_idx != word_dict['<pad>']       # 만약 <pad>면 제외
                                           and word_idx != word_dict['<eos>']])     # 만약 <eos>면 제외
                                                
                        caps_word.append(' '.join([convert_word_dict[word_idx] for word_idx in caption  # METEOR에 사용할 word로 구성된 Ground Truth value 수집 
                                                   if word_idx != word_dict['<start>']                  # 만약 <start>면 제외
                                                   and word_idx != word_dict['<pad>']                   # 만약 <pad>면 제외
                                                   and word_idx != word_dict['<eos>']]))                # 만약 <eos>면 제외
                        
                    references_word.append(caps_word[0])    # 5개의 Ground Truth value(captions)중 하나만 받음 (5개 모두 받으면 기하급수적으로 커진다.)
                    references_token.append(caps_token)     # 5개의 Ground Truth value(captions) list로 저장
                
                if batch_idx % 1000 == 0:
                    meteor = metric_meteor.compute(predictions=hypotheses_word, references=references_word)
                    bleu_4 = corpus_bleu(references_token, hypotheses_token)
                    print(f'BLEU-4: {bleu_4*100:.2f}, METEOR: {meteor["meteor"]*100:.2f}')
                
        meteor = metric_meteor.compute(predictions=hypotheses_word, references=references_word)
        bleu_4 = corpus_bleu(references_token, hypotheses_token)
        print(f'result = BLEU-4: {bleu_4*100:.2f}, METEOR: {meteor["meteor"]*100:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning project')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--data', type=str, default='data/coco',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--checkpoint', type=str, default='./model/model_vit_4_new_1e-5.pth',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161', 'vit'], default='vit',
                    help='Network to use in the encoder (default: vgg19)')
    
    eval(parser.parse_args())