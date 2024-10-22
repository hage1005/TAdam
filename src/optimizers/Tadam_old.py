import torch
import numpy
import wandb
import torch.nn as nn

class TadamOld(torch.optim.Optimizer):
    def __init__(self, params, total_steps, lr=1e-3, beta1=0.9, beta2=0.999, gamma=0.25, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, gamma=gamma, eps=eps, total_steps=total_steps)
        super(TadamOld, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params = group['params']
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            gamma = group['gamma']
            eps = group['eps']
            total_steps = group['total_steps']
            
            # Initialize time step, bias correction terms, and trust region parameters
            if 't' not in group:
                group['t'] = 1
                group['m_h'] = []
                group['m'] = []
                group['s'] = []
                group['v'] = []
                group['bp1'] = beta1
                group['bp2'] = beta2
                group['ls_h'] = 0.0
                group['ls'] = 0.0
                group['pr'] = 0.0
                group['dt'] = 1.0
                
                # Initialize momentum and variance buffers
                for p in params:
                    if p.requires_grad:
                        group['m_h'].append(torch.zeros_like(p.data))
                        group['m'].append(torch.zeros_like(p.data))
                        group['s'].append(torch.zeros_like(p.data))
                        group['v'].append(torch.zeros_like(p.data))

            t = group['t']
            dt = group['dt']
            bp1 = group['bp1']
            bp2 = group['bp2']

            device = params[0].device

            # Bias correction
            bc1 = 1.0 - bp1
            bc2 = 1.0 - bp2

            pr_temp = 0.0
            for i, p in enumerate(params):
                if p.grad is None:
                    continue
                m = group['m'][i]
                s = group['s'][i]
                v = group['v'][i]
                m_h = group['m_h'][i]

                # Update first and second moment estimates
                m.mul_(beta1).add_(1 - beta1, p.grad)
                m_h.copy_(m / bc1)
                
                s.mul_(beta2).addcmul_(1 - beta2, p.grad, p.grad)
                s_h = s / bc2
                
                dv = torch.square(p.grad - m_h) * (beta2 - bp2) / bc2
                v.mul_(beta2).add_(1 - beta2, dv)
                v_h = v / bc2

                # Fisher approximation and trust region update
                f_h = (1.0 + torch.sum(torch.square(m_h) / (v_h + eps))) * v_h
                u_h = torch.max(dt * f_h, torch.sqrt(s_h))  # rearranged inverse and delta
                g_h = m_h * dt / (u_h + eps)

                # Apply update to parameters
                p.data.add_(-lr * g_h)
                
                pr1 = torch.sum(m_h * g_h)
                pr2 = torch.sum(v_h * g_h ** 2)
                pr_temp += (pr1 - 0.5 * pr2) * lr

            # Update running loss and predict reduction
            group['ls'] = beta1 * group['ls'] + (1.0 - beta1) * loss
            group['ls_h'] = group['ls'] / bc1
            pr1 = torch.sum(m_hs * g_hs)
            pr2 = torch.sum(v_hs * g_hs ** 2)
            group['pr'] = pr_temp

            # Update betas
            group['bp1'] *= beta1
            group['bp2'] *= beta2

            # Compute new delta (dt)
            rho = (group['ls_h'] - loss) / max(group['pr'], eps)
            wandb.log({'rho': rho})
            dt_min = (1.0 - gamma) ** ((t - 1) / total_steps)
            dt_max = 1.0 + gamma ** ((t - 1) / total_steps)
            if rho < gamma:
                dt = dt_min
            elif rho > 1.0 - gamma:
                dt = dt_max
            else:
                dt = 1.0
            group['dt'] = max(min(dt * dt, dt_max), dt_min)

            # Increment time step
            group['t'] += 1

        return loss
