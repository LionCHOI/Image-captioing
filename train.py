# https://github.com/AaronCCWong/Show-Attend-and-Tell 
import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from include.dataset import ImageCaptionDataset
from include.decoder import Seq2SeqDecoder, create_mask
from include.encoder import Encoder
from include.utils import AverageMeter, accuracy, calculate_caption_lengths


# global variable --------------------------------------------------------------------------------------------------
data_transforms = transforms.Compose([                  # Dataloader에 사용할 transform 설정  
    transforms.Resize((224, 224)),                      # transform 설정 - size를 (224, 224)로 설정
    transforms.ToTensor(),                              # transform 설정 - tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # transform 설정 - Normalize - 평균 값 설정
                         std=[0.229, 0.224, 0.225])     # transform 설정 - Normalize - 표준편차 값 설정
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda 설정

PAD_IDX = 3 # PADDING index 
EOS_IDX = 1 # EOS index 


# MAIN =============================================================================================================
def main(args):
    # writer setting    ---------------------------------------------------------------------
    writer = SummaryWriter()    # tensorboard에 학습 결과를 update하고 기록해주는 부분

    # Data Loader   -------------------------------------------------------------------------
    word_dict = json.load(open(args.data + '/word_dict.json', 'r')) # vocab을 불러오는 부분
    vocabulary_size = len(word_dict)                                # vocab의 크기

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

    if args.model:                  
        decoder.load_state_dict(torch.load(args.model))         # 만약 미리 학습된 모델이 있으면 불러오기

    encoder.to(device)                                          # encoder model을 GPU에 넣기
    decoder.to(device)                                          # decoder model을 GPU에 넣기

    # training setting  ---------------------------------------------------------------------
    optimizer = optim.Adam(decoder.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)  # opimizer 설정 
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)                    # Scheduler 설정 (epoch이 작아서 코드상 쓰이지 않음)
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)                               # loss 함수 설정

    # data setting  -------------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(     # train loader 설정
        ImageCaptionDataset(data_transforms,        # Image Caption Dataset 설정 - transform 설정
                            args.data),             # Image Caption Dataset 설정 - data path를 넘겨줌
        batch_size=args.batch_size,                 # batch size 설정
        shuffle=True,                               # Shuffle 설정
        num_workers=1)                              # worker 수 설정

    val_loader = torch.utils.data.DataLoader(       # valid loader 설정
        ImageCaptionDataset(data_transforms,        # Image Caption Dataset 설정 - transform 설정
                            args.data,              # Image Caption Dataset 설정 - data path를 넘겨줌
                            split_type='val'),      # Image Caption Dataset 설정 - valid data로 설정
        batch_size=args.batch_size,                 # batch size 설정
        shuffle=False,                              # Shuffle 설정
        num_workers=1)                              # worker 수 설정

    # training start    -----------------------------------------------------------------------
    print('Starting training with {}'.format(args)) # train 시작 부분 터미널에 출력
    for epoch in range(1, args.epochs + 1):         # epoch만큼 반복문 시작 
        
        # train -------------------------------------------------------------------------------
        train(epoch=epoch,                              # 현재 epoch을 넘겨줌 (writer에서 사용할 예정)
              encoder=encoder,                          # encoder 넘겨줌   
              decoder=decoder,                          # decoder 넘겨줌
              optimizer=optimizer,                      # optimizer 넘겨줌
              scheduler=scheduler,                      # scheduler 넘겨줌
              cross_entropy_loss=cross_entropy_loss,    # cross_entropy_loss 넘겨줌
              data_loader=train_loader,                 # train_loader 넘겨줌       
              word_dict=word_dict,                      # vocab 넘겨줌
              log_interval=args.log_interval,           # 출력 간격을 넘겨줌 (해당 크기만큼 출력할 예정)
              writer=writer)                            # writer를 넘겨줌
        
        # valid --------------------------------------------------------------------------------
        validate(epoch=epoch,                           # 현재 epoch을 넘겨줌 (writer에서 사용할 예정)
                 encoder=encoder,                       # encoder 넘겨줌
                 decoder=decoder,                       # decoder 넘겨줌
                 cross_entropy_loss=cross_entropy_loss, # cross_entropy_loss 넘겨줌
                 data_loader=val_loader,                # val_loader 넘겨줌 
                 word_dict=word_dict,                   # vocab 넘겨줌
                 log_interval=args.log_interval,        # 출력 간격을 넘겨줌 (해당 크기만큼 출력할 예정)
                 writer=writer)                         # writer를 넘겨줌
        
        # model save    -------------------------------------------------------------------------
        model_file = 'model/model_' + args.network + '_' + str(epoch) + '.pth'  # 저장할 이름을 설정
        torch.save(decoder.state_dict(), model_file)                            # 저장 수행
        print('Saved model to ' + model_file)                                   # 저장 내역 터미널에 출력
    
    # training end  -----------------------------------------------------------------------------
    writer.close()  # tensorboard writer 종료


# TRAIN FUNCTION =============================================================================================================
def train(epoch, encoder, decoder, optimizer, scheduler, cross_entropy_loss, data_loader, word_dict, log_interval, writer):
    """ train 부분

    Args:
        epoch (int): 현재 epoch 값
        encoder (nn.module): encoder model
        decoder (nn.module): decoder model
        optimizer (Optimizer): optimizer
        scheduler (_LRScheduler): scheduler
        cross_entropy_loss (_WeightedLoss): cross_entropy_loss
        data_loader (DataLoader): train_loader
        word_dict (dict): vocabulary
        log_interval (int): 출력할 간격
        writer (object): writer
    """
    # model setting     --------------------------------------------------------------------------
    encoder.eval()  # encoder 부분을 eval()로 하여 학습하지 못하도록 한다.
    decoder.train() # decoder 부분만 학습하도록 한다.

    # loss 및 accuracy 지표 설정    ---------------------------------------------------------------
    losses = AverageMeter() # loss 변수
    top1 = AverageMeter()   # top 1 accuracy 변수 
    top5 = AverageMeter()   # top 5 accuracy 변수
    
    # training part ------------------------------------------------------------------------------
    for batch_idx, (imgs, captions) in enumerate(data_loader):  # for문 수행
        # data load & preprocessing --------------------------------------------------------------
        imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)   # data를 대입한다.
    
        tgt_input = captions[:, :-1]                                                # decoder input (크기를 맞추기 위해 마지막 부분 제거)
        targets = captions[:, 1:]                                                   # real value (loss에 활용할 데이터로 <SOS> 제거)    
        
        # processing    ----------------------------------------------------------------------------
        img_features = encoder(imgs)                                    # image를 encoder에 넣어 image feature 획득
        optimizer.zero_grad()                                           # opimizer zero_grad 설정
        
        tgt_mask, tgt_padding_mask = create_mask(tgt_input=tgt_input,   # masking for decoder - masking할 image captions
                                                 PAD_IDX=PAD_IDX)       # masking for decoder - padding index (padding 부분도 masking 하고자)
        
        preds = decoder(tgt_input=tgt_input,                            # decoder process - 학습에 사용할 image captions
                        img_features=img_features,                      # decoder process - 학습에 사용할 image features
                        tgt_mask=tgt_mask,                              # decoder process - 학습에 사용할 image captions mask
                        tgt_padding_mask=tgt_padding_mask)              # decoder process - 학습에 사용할 image captions padding mask
        
        loss = cross_entropy_loss(preds.reshape(-1, preds.shape[-1]),   # loss calcualtion - size 변경한 pred value [32, 50, 10366] --> [1600, 10366]
                                  targets.reshape(-1))                  # loss calcualtion - size 변경한 real value [32, 50]        --> [1600]
        
        loss.backward()                                                 # loss backpropagation
        optimizer.step()                                                # opimizer step

        # check performance ------------------------------------------------------------------------
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]     # pack_padded_sequence for pred value - acc 측정에서 빠른 연산을 위해 수행
        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0] # pack_padded_sequence for real value - acc 측정에서 빠른 연산을 위해 수행

        total_caption_length = calculate_caption_lengths(word_dict, captions)   # 총 길이 계산
        acc1 = accuracy(preds, targets, 1)                                      # top 1 accuracy 계산
        acc5 = accuracy(preds, targets, 5)                                      # top 5 accuracy 계산
        
        losses.update(loss.item(), total_caption_length)        # loss value update               
        top1.update(acc1, total_caption_length)                 # top 1 accuracy value update
        top5.update(acc5, total_caption_length)                 # top 5 accuracy value update

        if batch_idx % log_interval == 0:                                               # log interval에 해당하면 출력
            print('Train Batch: [{0}/{1}]\t'                                            # 현재 batch index 출력
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'                              # 현재 loss value 출력
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'                    # 현재 top 1 accuracy value 출력
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(              # 현재 top 5 accuracy value 출력
                      batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5)) 
    
    # scheduler step    ----------------------------------------------------------------------------- 
    scheduler.step()    # scheduler step
    
    # writing tensorboard   -------------------------------------------------------------------------
    writer.add_scalar('train_loss', losses.avg, epoch)      # tensorboard에 loss mean value 추가
    writer.add_scalar('train_top1_acc', top1.avg, epoch)    # tensorboard에 top1 mean value 추가
    writer.add_scalar('train_top5_acc', top5.avg, epoch)    # tensorboard에 top5 mean value 추가


# VALIDATION FUNCTION =============================================================================================================
def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, log_interval, writer):
    """ Validation 부분

    Args:
        epoch (int): 현재 epoch 값
        encoder (nn.module): encoder model
        decoder (nn.module): decoder model
        cross_entropy_loss (_WeightedLoss): cross_entropy_loss
        data_loader (DataLoader): val_loader
        word_dict (dict): vocabulary
        log_interval (int): 출력할 간격
        writer (object): writer
    """
    # model setting     --------------------------------------------------------------------------
    encoder.eval()  # encoder 부분을 eval()로 하여 학습하지 못하도록 한다.
    decoder.eval()  # decoder 부분을 eval()로 하여 학습하지 못하도록 한다.

    # loss 및 accuracy 지표 설정    ---------------------------------------------------------------
    losses = AverageMeter() # loss 변수
    top1 = AverageMeter()   # top 1 accuracy 변수
    top5 = AverageMeter()   # top 5 accuracy 변수

    # BLEU score를 위한 변수 설정   -----------------------------------------------------------------
    references = []     # real value를 위한 list
    hypotheses = []     # pred value를 위한 list
    
    # processing part ------------------------------------------------------------------------------
    with torch.no_grad():                                                           # gradient 계산하지 않도록 설정
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):    # for문 수행
            # data load & preprocessing --------------------------------------------------------------
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)   # data를 대입한다.
            
            tgt_input = captions[:, :-1]                                                # decoder input (크기를 맞추기 위해 마지막 부분 제거)
            targets = captions[:, 1:]                                                   # real value (loss에 활용할 데이터로 <SOS> 제거)    
        
            # processing    ----------------------------------------------------------------------------
            img_features = encoder(imgs)                                    # image를 encoder에 넣어 image feature 획득
            
            tgt_mask, tgt_padding_mask = create_mask(tgt_input=tgt_input,   # masking for decoder - masking할 image captions
                                                     PAD_IDX=PAD_IDX)       # masking for decoder - padding index (padding 부분도 masking 하고자)

            preds = decoder(tgt_input=tgt_input,                            # decoder process - 학습에 사용할 image captions
                            img_features=img_features,                      # decoder process - 학습에 사용할 image feature
                            tgt_mask=tgt_mask,                              # decoder process - 학습에 사용할 image captions mask
                            tgt_padding_mask=tgt_padding_mask)              # decoder process - 학습에 사용할 image captions padding mask
            
            loss = cross_entropy_loss(preds.reshape(-1, preds.shape[-1]),   # loss calcualtion - size 변경한 pred value [32, 50, 10366] --> [1600, 10366]
                                      targets.reshape(-1))                  # loss calcualtion - size 변경한 real value [32, 50]        --> [1600]

            # check performance ------------------------------------------------------------------------
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]  # pack_padded_sequence for pred value - acc 측정에서 빠른 연산을 위해 수행
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]     # pack_padded_sequence for real value - acc 측정에서 빠른 연산을 위해 수행
        
            total_caption_length = calculate_caption_lengths(word_dict, captions)   # 총 길이 계산
            acc1 = accuracy(packed_preds, targets, 1)                               # top 1 accuracy 계산
            acc5 = accuracy(packed_preds, targets, 5)                               # top 5 accuracy 계산
            
            losses.update(loss.item(), total_caption_length)    # loss value update 
            top1.update(acc1, total_caption_length)             # top 1 accuracy value update
            top5.update(acc5, total_caption_length)             # top 5 accuracy value update

                                                                        # real value ------------------------------------------
            for cap_set in all_captions.tolist():                       # BLEU를 위한 real captions을 reference list에 추가
                caps = []                                               # 여러 개의 real captions를 넣은 list
                for caption in cap_set:                                 # 각 real caption 문장마다 수행
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] # 만약 <start>면 제외
                                    and word_idx != word_dict['<pad>']  # 만약 <pad>면 제외
                                    and word_idx != word_dict['<eos>']] # 만약 <eos>면 제외
                    
                    caps.append(cap)                                    # 이렇게 생성된 real caption을 real captions list에 넣기
                references.append(caps)                                 # 여러 개의 real caption 문장들로 구성된 것을 넣기
                                                                        
                                                                        # pred value ------------------------------------------
            word_idxs = torch.max(preds, dim=2)[1]                      # pred value에서 가장 높은 값을 추출 
            for idxs in word_idxs.tolist():                             # BLEU를 위한 pred captions을 reference list에 추가
                hypotheses.append([idx for idx in idxs                  # 각 pred caption 문장마다 수행
                                       if idx != word_dict['<start>']   # 만약 <start>면 제외
                                       and idx != word_dict['<pad>']    # 만약 <pad>면 제외
                                       and idx != word_dict['<eos>']])  # 만약 <eos>면 제외
                

            if batch_idx % log_interval == 0:                                               # log interval에 해당하면 출력
                print('Validation Batch: [{0}/{1}]\t'                                       # 현재 batch index 출력
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'                              # 현재 loss value 출력
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'                    # 현재 top 1 accuracy value 출력
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(              # 현재 top 5 accuracy value 출력
                          batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))

        # check BLEU scoer -----------------------------------------------------------------------------
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))          # BLEU 1 score 계산 
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))      # BLEU 2 score 계산
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)) # BLEU 3 score 계산
        bleu_4 = corpus_bleu(references, hypotheses)                                # BLEU 4 score 계산
        
        print('Validation Epoch: {}\t'                                              # 현재 epoch index 출력
              'BLEU-1 ({})\t'                                                       # BLEU 1 score 계산
              'BLEU-2 ({})\t'                                                       # BLEU 2 score 계산
              'BLEU-3 ({})\t'                                                       # BLEU 3 score 계산
              'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))        # BLEU 4 score 계산
        
        # writing tensorboard   -------------------------------------------------------------------------
        writer.add_scalar('val_loss', losses.avg, epoch)    # tensorboard에 loss mean value 추가    
        writer.add_scalar('val_top1_acc', top1.avg, epoch)  # tensorboard에 top1 mean value 추가
        writer.add_scalar('val_top5_acc', top5.avg, epoch)  # tensorboard에 top5 mean value 추가

        writer.add_scalar('val_bleu1', bleu_1, epoch)       # tensorboard에 BLEU 1 score 추가
        writer.add_scalar('val_bleu2', bleu_2, epoch)       # tensorboard에 BLEU 2 score 추가
        writer.add_scalar('val_bleu3', bleu_3, epoch)       # tensorboard에 BLEU 3 score 추가
        writer.add_scalar('val_bleu4', bleu_4, epoch)       # tensorboard에 BLEU 4 score 추가

# main part =============================================================================================================
if __name__ == "__main__":
    # parser setting ----------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Image Captioning project')                            # parser 설정                    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',                              # batch size 설정 (default:32)
                        help='batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',                                   # epoch 설정 (default:5)
                        help='number of epochs to train for (default: 5)')                             
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',                                 # learning rate 설정 (default:5e-5)
                        help='learning rate of the decoder (default: 5e-5)')
    parser.add_argument('--step-size', type=int, default=5,                                             # step size 설정 (default:5)
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',                           # log-interval 설정 (default:100)
                        help='number of batches to wait before logging training stats (default: 100)')  
    parser.add_argument('--data', type=str, default='data/coco/max_caption_5',                          # data path 설정 (default:data/coco/max_caption_5)
                        help='path to data images (default: data/coco/max_caption_5)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161', 'vit'],              # encoder network 설정 (default:vit)
                        default='vit',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, help='path to model')                                      # check point 설정 (default:None)

    # main 수행 -----------------------------------------------------------------------------------------------------------
    main(parser.parse_args())
