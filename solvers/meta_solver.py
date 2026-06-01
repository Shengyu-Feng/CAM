import os
import time
import itertools
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from accelerate import Accelerator
from torch_geometric.loader import DataLoader as GDataLoader
from torch.utils.data import DataLoader
import importlib
from torch import nn
from utils.diffusion_schedulers import *
from utils.lr_schedulers import get_schedule_fn
from utils.temp_schedulers import TempScheduler
from utils.data_utils import *

# MetaSolver
class MetaSolver(object):
    def __init__(self, args):
        self.args = args

        self.num_t = args.num_t
        self.num_k = args.num_k
        self.num_repetition = args.repetition

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.skip_decode = args.skip_decode
        self.model = None

        module = importlib.import_module('.'.join(['datasets', args.problem]))
        dataset = getattr(module, 'Dataset')
        module = importlib.import_module('.'.join(['envs', args.problem]))
        problem = getattr(module, 'Env')
        self.problem = problem(args)

        self.solution_type = self.problem.solution_type

        if args.data_precision == 'fp32':
            self.dtype = torch.float32
        elif args.data_precision == 'fp16':
            self.dtype = torch.float16
        elif args.data_precision == 'bf16':
            # Note bf16 does not support kth value in RLSA
            self.dtype = torch.bfloat16
        else:
            raise NotImplementedError

        if args.do_train:
            train_dataset = dataset(args.train_path, self.dtype, args.train_samples, args)
            valid_dataset = dataset(args.valid_path, self.dtype, args.valid_samples, args)
        else:
            test_dataset = dataset(args.test_path, self.dtype, args.test_samples, args)

        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb")
        accelerator.init_trackers(
            project_name=args.wandb_name,
            config=vars(args),
            init_kwargs={"wandb": {"entity": args.wandb_entity or os.getenv("WANDB_ENTITY"), "name": args.wandb_run_name}}
            )
        self.accelerator = accelerator

        if args.do_train:
            self.train_loader = GDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=False)
            self.valid_loader = GDataLoader(valid_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
            self.test_loader = None
        else:
            self.test_loader = GDataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
            self.train_loader = None
            self.valid_loader = None
            
    def prepare(self):
        self.model, self.optimzier, self.scheduler, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_loader, self.valid_loader, self.test_loader)
        if getattr(self.args, 'compile', False):
            self.accelerator.print("Compiling model with torch.compile...")
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model)
        
    def log(self, logger):
        self.accelerator.log(logger, step=self.global_step)
        
    def print_log(self, logger):
        log = []
        for k, v in logger.items():
            log.extend([k + ':', str(v)])
        log = ' '.join(log)
        self.accelerator.print(log)

    def evaluate_single(graph):
        raise NotImplementedError
    
    @torch.no_grad()
    def evaluate(self, mode, log_path=None):
        assert mode in ['valid', 'test']
        if mode == 'valid':
            dataloader = self.valid_loader
        else:
            dataloader = self.test_loader
        if self.model is not None:
            self.model.eval()

        all_time = []   
        all_eval_set_size = []
        
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not self.accelerator.is_local_main_process):
            st_time = time.time()
            eval_set_size = self.evaluate_single(batch)
            end_time = time.time()
            all_time.append(torch.tensor(end_time-st_time, device=self.accelerator.device))
            all_eval_set_size.append(eval_set_size)

        self.accelerator.wait_for_everyone()

        all_time = torch.stack(all_time)

        all_time = self.accelerator.gather(all_time)

        all_eval_set_size = torch.stack(all_eval_set_size) 
        all_eval_set_size = self.accelerator.gather(all_eval_set_size)  

        obj_result = torch.mean(all_eval_set_size).item()
        time_result = torch.mean(all_time).item()
        
        if log_path is not None:
            if self.accelerator.is_local_main_process:
                self.accelerator.print('Avg. obj: %f\n'%obj_result, 'Avg. time: %f\n'%time_result)
                with open(log_path, 'w') as f:
                    f.write('Avg. obj: %f\n'%obj_result)
                    f.write('Avg. time: %f\n'%time_result)
        return obj_result, time_result

# Neural Solver
class NNSolver(MetaSolver):
    def __init__(self, args):
        super(NNSolver, self).__init__(args)
        
        self.time_scale = args.time_scale
        self.num_tp = args.num_tp
        if args.method == 'DIFUSCO':
            self.time_scale = 1
        module = importlib.import_module('models')
        model = getattr(module, args.model)
        self.model = model(
            n_layers=args.num_l,
            hidden_dim=args.num_h,
            out_channels=2 if args.method=='DIFUSCO' else 1,
            use_activation_checkpoint=args.use_activation_checkpoint,
            problem=args.problem,
            time_scale=self.time_scale
        )
            
        if args.do_train:
            self.num_kp = args.num_kp
            self.batch_size = args.batch_size
            self.mini_batch_size = args.mini_batch_size
            self.batches_per_step = args.batches_per_step
            self.global_step = 0

            self.grad_clip = args.grad_clip
            self.patience = args.patience

            self.epochs = args.epochs
            self.inner_loop_steps = args.inner_loop_steps
            self.iterations_per_epoch = int(args.samples_per_epoch // self.accelerator.num_processes // self.batch_size)
            self.configure_optimizer(args)
            if args.checkpoint is not None:
                state_dict = torch.load(args.checkpoint, weights_only=True, map_location=self.accelerator.device)
                self.load_ckpt(state_dict)
                self.accelerator.print(f"Resumed from checkpoint: {args.checkpoint}")
        else:
            ckpt_path = args.checkpoint if args.checkpoint is not None else os.path.join(args.save_dir, 'best.pt')
            state_dict = torch.load(ckpt_path, weights_only=True, map_location=self.accelerator.device)
            self.load_ckpt(state_dict)
      
    def configure_optimizer(self, args):
        try:
            OptimCls = getattr(optim, args.optimizer)
        except AttributeError:
            raise NotImplementedError(f"Unknown optimizer: {args.optimizer}")

        self.optimizer = OptimCls(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        num_steps = self.num_steps()
        self.scheduler = get_schedule_fn(args.lr_scheduler, num_steps)(self.optimizer)
            
    def load_ckpt(self, ckpt):
        self.model.load_state_dict(ckpt)

    def save_ckpt(self, save_file):
        if self.accelerator.is_local_main_process:
            model_to_save = self.accelerator.unwrap_model(self.model)

            if hasattr(model_to_save, "_orig_mod"):
                model_to_save = model_to_save._orig_mod

            torch.save(model_to_save.state_dict(), save_file)
    
    def train_batch(batch):
        raise NotImplementedError
    
    def train(self, save_dir): 
        raise NotImplementedError     
            
                
# Supervised Solver
class SLSolver(NNSolver):
    def __init__(self, args):
        super(SLSolver, self).__init__(args)
                 
    def num_steps(self):
        return len(self.train_loader)*self.epochs
            
    def train(self, save_dir):
        best_eval = np.inf
        wait = 0
        self.accelerator.wait_for_everyone()

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for _, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=not self.accelerator.is_local_main_process):
                
                loss = self.train_batch(batch)     
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip
                    )
                self.optimizer.step()  
                self.scheduler.step()
                
                self.global_step += 1
                losses.append(loss.detach())
            logger = {'Epoch': epoch, 'train/loss': torch.stack(losses).mean().item()}
            self.log(logger)

            self.accelerator.wait_for_everyone() 
            
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            
            logger = {'Epoch': epoch, 'eval/object': obj_result, 'eval/time': time_result}
            
            self.log(logger)
            self.print_log(logger)
            
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
                wait = 0
            else:
                wait += 1
                
            if wait == self.patience:
                self.accelerator.print('Early stop')
                break

# Unsupervised solver
class ULSolver(NNSolver):
    def __init__(self, args):
        super(ULSolver, self).__init__(args)
                
        if args.do_train:
            self.tau = args.tau0
            assert args.epochs >= args.N_warmup + args.N_anneal, "Total epochs must be at least the sum of warmup and anneal steps."
            self.temp_scheduler = TempScheduler(args.temp_schedule, args.tau0, args.tau1, args.N_warmup, args.N_anneal, args.epochs - args.N_warmup - args.N_anneal)    

    def num_steps(self):
        if self.batches_per_step is None:
            batches_per_step = self.full_batches_per_step()
        else:
            batches_per_step = self.batches_per_step
        return self.epochs*self.iterations_per_epoch*self.inner_loop_steps*batches_per_step

    @torch.no_grad()
    def generate_data(self, batch):
        raise NotImplementedError
    
    def train(self, save_dir):
        best_eval = np.inf
        wait = 0
        self.accelerator.wait_for_everyone()

        for epoch in range(self.epochs):
            sample_loader = iter(cycle(self.train_loader))
            self.model.train()
            self.tau = self.temp_scheduler(epoch)
            losses = []

            for _ in tqdm(range(self.iterations_per_epoch), total=self.iterations_per_epoch, disable=not self.accelerator.is_local_main_process):
                graph = next(sample_loader)

                if self.solution_type == 'node':
                    A = torch.sparse_coo_tensor(
                        graph.edge_index,
                        graph.edge_weight,
                        torch.Size((graph.num_nodes, graph.num_nodes))
                    ).to_sparse_csr()
                    graph.A = A

                self.model.eval()
                batch_set = self.generate_data(graph)
                batch_loader = DataLoader(batch_set, batch_size=self.mini_batch_size, shuffle=True)

                self.model.train()
                for _ in range(self.inner_loop_steps):
                    for batch in itertools.islice(batch_loader, self.batches_per_step):
                        loss = self.train_batch(graph, batch)
                        self.optimizer.zero_grad()
                        self.accelerator.backward(loss)

                        if self.grad_clip:
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.grad_clip
                            )

                        self.optimizer.step()
                        self.scheduler.step()
                        self.global_step += 1

                        losses.append(loss.detach())
                    
            logger = {'Epoch': epoch, 'train/loss': torch.stack(losses).mean().item()}
            self.log(logger)
            self.accelerator.wait_for_everyone() 
            
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            
            logger = {'Epoch': epoch, 'eval/object': obj_result, 'eval/time': time_result}
            self.log(logger)
            self.print_log(logger)
            
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
                wait = 0
            else:
                wait += 1
                
            if wait == self.patience:
                self.accelerator.print('Early stop')
                break
                
    def evaluate_single(self, graph):
        best_energies = []
        for _ in range(self.num_repetition):
            if self.solution_type == 'edge':
                num_edges = graph.edge_index.shape[1]
                solution = torch.randint(0,2,size=(num_edges,self.num_k), device=self.accelerator.device, dtype=torch.float)
            else:
                num_nodes = graph.num_nodes
                solution = torch.randint(0,2,size=(num_nodes,self.num_k), device=self.accelerator.device, dtype=torch.float)
                A = torch.sparse_coo_tensor(
                    graph.edge_index, 
                    graph.edge_weight, 
                    torch.Size((graph.num_nodes, graph.num_nodes))
                ).to_sparse_csr()
                graph.A = A
            
            for t in range(self.num_t):     
                time = torch.ones_like(solution)*int(t*self.num_tp/self.num_t)/self.num_tp
                logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
                flip_prob = torch.sigmoid(logits)
                if t == self.num_t-1:
                    solution = solution*(1-flip_prob) + flip_prob*(1-solution)
                else:
                    rr = torch.rand_like(solution)
                    solution = torch.where(rr<flip_prob, 1-solution, solution)
            
            if not self.skip_decode:
                best_energy, _ = self.problem.evaluate(graph, solution)
            else:
                best_energy, _ = self.problem.energy_func(graph, solution)

            best_energies.append(best_energy)
            
        return torch.cat(best_energies).min()
    
                