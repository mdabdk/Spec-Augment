import numpy as np

"""
The following list contains the proportions of the total number of augmentations
that are time warped (TW), time masked (TM), and frequency masked (FM). These
transformations are considered as random variables with associated probability
distributions P(TW), P(TM), and P(FM) respectively. These are Bernoulli random
variables, where each transformation can either be applied (1) or not applied
(0).

The first element of the list corresponds to the proportion of the total number
of augmentations that are time warped, or P(TW = 1). The second element of the
list corresponds to the proportion of the total number of augmentations that are
time masked, or P(TM = 1). The third element of the list corresponds to the
proportion of the total number of augmentations that are frequency masked, or
P(FM = 1). Note that:

P(TW = 0) = 1 - P(TW = 1)
P(TM = 0) = 1 - P(TM = 1)
P(FM = 0) = 1 - P(FM = 1)

NOTE: Because the probability of not time warping, not time masking, and not
frequency masking a spectrogram is set to 0, then the sum of these
probabilities MUST be at least 1.5. In other words, sum(p) > 1.5.
"""

p = [0.0,0.9,0.9]

"""
The possible transformations that can be applied are shown in the table below.
TW stands for time warping, TM stands for time masking, and FM stands for
frequency masking. A 1 indicates that the corresponding transformation is
applied, while a 0 indicates that the corresponding transformation is not
applied.

TW  TM  FM
-----------
0   0   0
0   0   1
0   1   0
0   1   1
1   0   0
1   0   1
1   1   0
1   1   1

Since these transformations can be applied in combinations, then a joint
distribution for these transformations P(TW,TM,FM) also exists. However, these
transformations are assumed to be independent from each other, such that the
occurence of one transformation does not imply anything about the occurence of
another transformation. This means that:

P(TW,TM,FM) = P(TW) x P(TM) x P(FM)

To avoid no augmentations occuring at all, which is when TW = 0 and TM = 0 and
FM = 0, then the joint probability of these events should be set to 0:
    
P(TW = 0,TM = 0,FM = 0) = 0

However, the probability mass associated with P(TW = 0,TM = 0,FM = 0) must now
be re-distributed to the rest of the probability distribution. Since there
are 7 outcomes in total, then the following probability mass is re-distributed:
    
P(TW = 0,TM = 0,FM = 0) / 7 = P(TW = 0) x P(TM = 0) x P(FM = 0) / 7

This is equivalent to re-distributing:
    
(1 - P(TW = 1)) x (1 - P(TM = 1)) x (1 - P(FM = 1)) / 7

Re-distribution is done by adding this probability mass to the probability mass
of each of the other outcomes.
"""

redistribution = (1-p[0])*(1-p[1])*(1-p[2])/7

# the joint distribution P(TW,TM,FM) = P(TW) x P(TM) x P(FM) is defined here.
# Note that the sum of all probability masses must be equal to 1

joint = [0, # P(TW = 0,TM = 0,FM = 0)
         (1-p[0])*(1-p[1])*( p[2] ) + redistribution, # P(TW = 0,TM = 0,FM = 1)
         (1-p[0])*( p[1] )*(1-p[2]) + redistribution, # P(TW = 0,TM = 1,FM = 0)
         (1-p[0])*( p[1] )*( p[2] ) + redistribution, # P(TW = 0,TM = 1,FM = 1)
         ( p[0] )*(1-p[1])*(1-p[2]) + redistribution, # P(TW = 1,TM = 0,FM = 0)
         ( p[0] )*(1-p[1])*( p[2] ) + redistribution, # P(TW = 1,TM = 0,FM = 1)
         ( p[0] )*( p[1] )*(1-p[2]) + redistribution, # P(TW = 1,TM = 1,FM = 0)
         ( p[0] )*( p[1] )*( p[2] ) + redistribution] # P(TW = 1,TM = 1,FM = 1)

# initialize random number generator

rng = np.random.default_rng()

# the joint distribution is a discrete categorical distribution
# sample num_samples times from the categorical distribution

num_samples = 10000

x = rng.multinomial(n = 1,
                    pvals = joint, # event probabilities
                    size = num_samples) # number of augmentations

# x is in one-hot encoding, so obtain the actual numbers

x = np.nonzero(x)[1]

# check proportions/probability estimates

# time warping occurs when x = 4,5,6,7 (1,0,0 - 1,0,1 - 1,1,0 - 1,1,1)

p_hat0 = (np.sum(x==4) + 
          np.sum(x==5) + 
          np.sum(x==6) + 
          np.sum(x==7))/num_samples

# time masking occurs when x = 2,3,6,7 (0,1,0 - 0,1,1 - 1,1,0 - 1,1,1)

p_hat1 = (np.sum(x==2) + 
          np.sum(x==3) + 
          np.sum(x==6) + 
          np.sum(x==7))/num_samples

# frequency masking occurs when x = 1,3,5,7 (0,0,1 - 0,1,1 - 1,0,1 - 1,1,1)

p_hat2 = (np.sum(x==1) + 
          np.sum(x==3) + 
          np.sum(x==5) + 
          np.sum(x==7))/num_samples

# print results

for i,p_hat in enumerate([p_hat0,p_hat1,p_hat2]):
    print('Actual probability = {}'.format(p[i]))
    print('Estimated probability = {}'.format(p_hat))
    print('-'*30)
