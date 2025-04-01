import numpy as np

class kFoldGenerator_train():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y
        
        ## The validation set order is shifted by a random vector
        # val_shift = np.random.randint(0, self.k-1, self.k)
        # self.val_idx = (np.arange(self.k) + val_shift) % self.k
        
        ## Here, to make the result reproducible, we use the following fixed vector
        self.val_idx = [7, 6, 4, 2, 0, 9, 1, 0, 4, 4]

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p == self.val_idx[i]:
                val_data = self.x_list[p]
                val_targets = self.y_list[p]
                print('  Fold #', p, ': val')
            elif p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
                print('  Fold #', p, ': train')
        return train_data, np.argmax(train_targets,1), val_data, np.argmax(val_targets,1)


class kFoldGenerator_test():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y

    # Get i-th fold
    def getFold(self, i):
        test_data = self.x_list[i]
        test_targets = self.y_list[i]
        return test_data, np.argmax(test_targets,1)


class DomainGenerator_train():
    '''
    Domain Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domain
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = len_list
        self.k = len(len_list)
        
        # The validation set order is shifted by a random vector
        # Here, to make the result reproducible, we use the following fixed vector
        self.val_idx = [7, 6, 4, 2, 0, 9, 1, 0, 4, 4]

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        j = 0   #1~9
        ii = 0  #1~10
        for l in self.l_list:
            if ii == self.val_idx[i]:
                val_domain = np.zeros((l, self.k-1), dtype=int)
                print('  Fold #', ii, ': val')
            elif ii != i:
                a = np.zeros((l, self.k-1), dtype=int)
                a[:, j] = 1
                if isFirst:
                    train_domain = a
                    isFirst = False
                else:
                    train_domain = np.concatenate((train_domain, a))
                j += 1
                print('  Fold #', ii, ': train')
            ii += 1
        return np.argmax(train_domain, 1), np.argmax(val_domain, 1)


class DomainGenerator_test():
    '''
    Domain Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domain
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = len_list
        self.k = len(len_list)

    # Get i-th fold
    def getFold(self, i):
        # for test, all domain labels are 0
        test_domain = np.zeros((self.l_list[i], 9), dtype=int)
        return np.argmax(test_domain, 1)
