import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn



def cosine_sim(im, s, t=1):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s, t=1):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s, t=1):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score    

def exponential_sim(im, s, t=1):
    # need to check dimention matching
    return torch.exp(cosine_sim(im, s)/t)


class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        elif measure == 'exp':
            self.sim = exponential_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s, im):
        """
            s.shape = (128, 2048)
            im.shape = (128, 2048)

            Colab with step running: https://colab.research.google.com/drive/1sjW0Eo1zJYbiopXShf186NoKk7GHUzGd?usp=sharing
        """
        # compute image-sentence score matrix
        print("shape of sentence: {}\nshape of image: {}".format(s.shape, im.shape))
        
        scores = self.sim(im, s)
        # after sim: scores.shape = (128, 128)
        
        print("shape of scores: {}".format(scores.shape))

        # get the diagonal of the similiarty matrix
        diagonal = scores.diag().view(im.size(0), 1)
        # diagonal.shape = (128, 1)
        # Guess: scores[i][i] = pos score? Indeed.
        # TODO: Change the contrastive loss w.r.t this logic

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        # generate a binary matrix with the diagonal is True while the rest is False

        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, measure='exp', neg_sampling='random', direction='all',dataset='msrvtt', num_of_pairs=20,start_idx):
        super(ContrastiveLoss, self).__init__()
        """ margin: the margin used to select negative samples (see the Negative Sampling Methods slides)
            measure: how to compute similiarity
            neg_sampling: 'random', 'progressive': from easy to hard
            cost_style: used to decide how to add up sentence and image loss (sum or avg)
            direction: 'i2t' image to text retrieval, 't2i' text to image retrieval, 'all': both
            neg_n: number of negative samples
        """
        
        print(">"*20)
        print("Contrastive Loss Used")
        #self.margin = margin
        #self.cost_style = cost_style
        self.direction = direction
        self.neg_sampling = neg_sampling
        #could be a number(msrvtt) or a list of numbers indicating the number of positive pairs of one clip
        self.num_of_pairs = num_of_pairs
        self.dataset = dataset
        
        if dataset == 'tempuckey':
            self.start_idx = start_idx
        else:
            self.start_idx = list(range(0,120,self.num_of_pairs))
            
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        elif measure == 'exp':
            self.sim = exponential_sim
        else:
            self.sim = cosine_sim

    def forward(self, s, im, temperature=1, alpha=0):
        """
            s: a 2d tensor with a shape of (batch_size, feature_size) Note: for original dual encoder, it is (batch_size, 2048)
            im: a 2d tensor with a shape of (batch_size, feature_size) Note: for original dual encoder, it is (batch_size, 2048)
            label: a 1d binary list stands the relativeness of a video-text pair (1 = pos, 0 = not-pos)
            tempurature: used for simliarity
            alpha: used for negative sampling
        """
        # scores.shape = (batch_size, batch_size)
        scores = self.sim(im, s, t=temperature)
        batch_size = 128
        mask = np.zeros([batch_size,batch_size])
               
        #this mask can handle both tempuckey and msrvtt
        n = start_idx[1:]
        idx = zip(start_idx,n)
        for i,j in idx:
          mask[i:j,i:j] = 1                
            
        m_match = torch.Tensor(mask) == 0
        m_cost = torch.Tensor(mask) == 1
        Imatch = Variable(m_match)
        Icost = Variable(m_cost)
        
        if torch.cuda.is_available():
            Imatch = Imatch.cuda()
            Icost = Icost.cuda()

        cost_s = None
        cost_im = None
        match_s = None
        match_im = None
        
        # Implement negative sampling here
        # TODO !!!
        #MAY BE USE A MARGIN????
        if self.neg_sampling == 'all':
            if self.direction in  ['i2t', 'all']:
                # caption retrieval
                cost_s = scores.clamp(min=0)
                cost_s = cost_s.masked_fill_(Imatch, 0)
                match_s = scores.clamp(min=0)
                match_s = match_s.masked_fill_(m_cost, 0)
                
            if self.direction in ['t2i', 'all']:
                # image retrieval
                cost_im = scores.t().clamp(min=0)
                cost_im = cost_im.masked_fill_(Imatch, 0)
                match_im = scores.t().clamp(min=0)
                match_im = match_im.masked_fill_(m_cost, 0) 
        elif self.neg_sampling == 'progressive':
            raise NotImplementedError

        elif self.neg_sampling == 'random':
            raise NotImplementedError

        
        # Sum up and return
        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
            match_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()
            match_im = Variable(torch.zeros(1)).cuda()        
        #MIL-NCE loss
        return (cost_s.sum()+cost_im.sum()) / (cost_s.sum()+cost_im.sum() + match_s.sum() + match_im.sum())
        
