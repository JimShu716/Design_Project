import torch
from torch.autograd import Variable
import torch.nn as nn



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score    

def exponential_sim(im, s, t=1):
    # need to check dimention matching
    return torch.exp(torch.sum(im*s, 2))


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
        # compute image-sentence score matrix
        print("shape of sentence: {}\nshape of image: {}".format(s.shape, im.shape))
        scores = self.sim(im, s)
        print("shape of scores: {}".format(scores.shape))
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
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
    def __init__(self, margin=0, measure='exp', neg_sampling='random', cost_style='sum', direction='all', neg_n=10):
        super(ContrastiveLoss, self).__init__()
        """ margin: the margin used to select negative samples (see the Negative Sampling Methods slides)
            measure: how to compute similiarity
            neg_sampling: 'random', 'progressive': from easy to hard
            cost_style: used to decide how to add up sentence and image loss (sum or avg)
            direction: 'i2t' image to text retrieval, 't2i' text to image retrieval, 'all': both
            neg_n: number of negative samples
        """
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        self.neg_sampling = neg_sampling
        self.neg_n = neg_n
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        elif measure == 'exp':
            self.sim = exponential_sim
        else:
            self.sim = cosine_sim


    def forward(self, s, im, label, temperature, alpha):
        """
            s: a 3d tensor stands for sentence encoding
                i.e.
                    [
                        [[aaa],[aaa],[aaa],[aaa],[aaa],[aaa],[aaa]], - encoding of 7 clips from video 1
                        [[bbb],[bbb],[bbb],[bbb],[bbb],[bbb],[bbb]], - encoding of 7 clips from video 2
                        [[ccc],[ccc],[ccc],[ccc],[ccc],[ccc],[ccc]], - encoding of 7 clips from video 3
                    ]
            im: a 3d tensor stands for image/frame encoding
                i.e.
                    [
                        [[ddd],[ddd],[ddd],[ddd],[ddd],[ddd],[ddd]], - encoding of 7 piece of captions from video 1
                        [[eee],[eee],[eee],[eee],[eee],[eee],[eee]], - encoding of 7 piece of captions from video 2
                        [[fff],[fff],[fff],[fff],[fff],[fff],[fff]], - encoding of 7 piece of captions from video 3
                    ]
            label: a 2d binary list stands for if the video-sentence pair matchs to each other. 
                    0 - not match/not-pos pair 
                    1 - match/pos pair
                    i.e.
                        [
                            [0,0,1,1,1,0,0], - video 1
                            [0,1,1,1,0,0,0], - video 2
                            [0,0,0,0,1,1,0], - video 3
                        ]
            temperature: used for calculating similiarity 
        """

        # Step 1: Compute the sim score of all pairs
        scores = self.sim(im, s, t=temperature)

        # Step 2: Compute the sum of score of positive pairs
        label = torch.tensor(label)
        pos_scores = scores * label
        pos_scores_no_zero = list()
        for i in range(pos_scores.shape[0]):
            pos_scores_no_zero.append(pos_scores[i][torch.nonzero(pos_scores[i])].squeeze(-1))

        least_pos_scores = list()
        for i in pos_scores_no_zero:
            least_pos_scores.append(torch.min(i))

        sum_pos_scores = pos_scores.sum(1)

        # Step 3: Rank the sim score in decending order (suppose larger sim score == most similiar)
        # compute by sim(pos) - alpha - sim(others)
        # TBC
        score_rank = scores
        if self.neg_sampling == 'progressive':
            score_rank = -score_rank - alpha
            torch.add(score_rank, least_pos_scores)
            score_rank = torch.sort(score_rank).values

        # Step 4: Select positive and negative pairs
        num_pos = label.sum(1)
        sum_neg_scores = list()
        if self.neg_sampling == 'random':
            for idx, i in enumerate(score_rank):
                while(t<num_pos[idx] or t<self.neg_n):
                    raise NotImplementedError
                        
        elif self.neg_sampling == 'progressive':
            # use the rank sampling
            neg_sample_num = self.neg_n
            for rank in score_rank:
                sum_neg_scores.append(rank[0:neg_sample_num].sum())

        else:
            raise NotImplementedError

        # Step 5: Construct the loss
        loss = torch.zeros(s.shape[0])
        sum_neg_scores = torch.tensor(sum_neg_scores)
        sum_pos_scores = torch.tensor(sum_pos_scores)
        loss += sum_pos_scores/(sum_pos_scores+sum_neg_scores)
        return loss
