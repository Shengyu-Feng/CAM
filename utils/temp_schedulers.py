class TempScheduler(object):
    def __init__(self, schedule, tau0, tau1, N_warmup, N_anneal, N_equil):
        if schedule == 'linear':
            self.temp_scheduler = self.__linear_annealing
        elif schedule == 'exp':
            self.temp_scheduler = self.__exp_annealing
        else:
            raise NotImplementedError

        self.N_warmup = N_warmup
        self.N_anneal = N_anneal
        self.epochs = N_warmup + N_anneal + N_equil
        self.tau0 = tau0
        self.tau1 = tau1

    def __call__(self, epoch):
        return self.temp_scheduler(epoch)

    def __linear_annealing(self, epoch):
        if epoch < self.N_warmup:
            tau = self.tau0
        elif epoch >= self.N_warmup + self.N_anneal:
            tau = self.tau1
        else:
            tau = max([self.tau0 - self.tau1 - (self.tau0-self.tau1) * (epoch - self.N_warmup) / self.N_anneal, 0]) + self.tau1
        
        return tau 
    
    def __exp_annealing(self, epoch):
        if epoch < self.N_warmup:
            tau = self.tau0
        elif epoch >= self.N_warmup + self.N_anneal:
            tau = self.tau1
        else:
            factor = 4000
            tau = self.tau1/(1- 0.998**(factor*((epoch - self.N_warmup +1 )/self.epochs )))
        return tau 